# Testing code for document chat functionality

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present without overriding
# existing environment values. This ensures credentials can be supplied via
# either the shell or a local .env during development.
load_dotenv()

# Ensure LangSmith tracing variables are present for this test run.
# If not explicitly provided by the user/shell, enable tracing by default.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# LANGSMITH_API_KEY will be picked up from the environment or .env if set.

# import sys
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from src.single_document_chat.data_ingestion import SingleDocIngestor
# from src.single_document_chat.retrieval import ConversationalRAG
# from utils.model_loader import ModelLoader

# FAISS_INDEX_PATH = Path("faiss_index")

# def test_conversational_rag_on_pdf(pdf_path:str, question:str):
#     try:
#         model_loader = ModelLoader()
        
#         if FAISS_INDEX_PATH.exists():
#             print("Loading existing FAISS index...")
#             embeddings = model_loader.load_embeddings()
#             vectorstore = FAISS.load_local(folder_path=str(FAISS_INDEX_PATH), embeddings=embeddings,allow_dangerous_deserialization=True)
#             retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#         else:
#             # Step 2: Ingest document and create retriever
#             print("FAISS index not found. Ingesting PDF and creating index...")
#             with open(pdf_path, "rb") as f:
#                 uploaded_files = [f]
#                 ingestor = SingleDocIngestor()
#                 retriever = ingestor.ingest_files(uploaded_files)
                
#         print("Running Conversational RAG...")
#         session_id = "test_conversational_rag"
#         rag = ConversationalRAG(retriever=retriever, session_id=session_id)
#         response = rag.invoke(question)
#         print(f"\nQuestion: {question}\nAnswer: {response}")
                    
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
#         sys.exit(1)
    
# if __name__ == "__main__":
#     # Example PDF path and question
#     pdf_path = "data\\single_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
#     question = "What is the significance of the attention mechanism? can you explain it in simple terms?"

#     if not Path(pdf_path).exists():
#         print(f"PDF file does not exist at: {pdf_path}")
#         sys.exit(1)
    
#     # Run the test
#     test_conversational_rag_on_pdf(pdf_path, question)
    
    
## testing for multidoc chat
# import sys
# from pathlib import Path
# from src.multi_document_chat.data_ingestion import DocumentIngestor
# from src.multi_document_chat.retrieval import ConversationalRAG

# def test_document_ingestion_and_rag():
#     try:
#         test_files = [
#             "data\\multi_doc_chat\\market_analysis_report.docx",
#             "data\\multi_doc_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf",
#             "data\\multi_doc_chat\\sample.pdf",
#             "data\\multi_doc_chat\\state_of_the_union.txt"
#         ]
        
#         uploaded_files = []
        
#         for file_path in test_files:
#             if Path(file_path).exists():
#                 uploaded_files.append(open(file_path, "rb"))
#             else:
#                 print(f"File does not exist: {file_path}")
                
#         if not uploaded_files:
#             print("No valid files to upload.")
#             sys.exit(1)
            
#         ingestor = DocumentIngestor()
        
#         retriever = ingestor.ingest_files(uploaded_files)
        
#         for f in uploaded_files:
#             f.close()
                
#         session_id = "test_multi_doc_chat"
        
#         rag = ConversationalRAG(session_id=session_id, retriever=retriever)
        
#         question = "what is President Zelenskyy said in their speech in parliament?"
        
#         answer=rag.invoke(question)
        
#         print("\n Question:", question)
        
#         print("Answer:", answer)
        
#         if not uploaded_files:
#             print("No valid files to upload.")
#             sys.exit(1)
            
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
#         sys.exit(1)
        
# if __name__ == "__main__":
#     test_document_ingestion_and_rag()




import sys
import os
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage

def test_document_ingestion_and_rag():
    try:
        test_files = [
            "/Users/yashpatil/Developer/AI/YT/Sunny/LLMOps_series/data/multi_doc_chat/NIPS-2017-attention-is-all-you-need-Paper.pdf",
        ]

        uploaded_files = []

        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

        # Build index using single-module ChatIngestor
        ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
        # Option 1: Use similarity search (default)
        # _ = ci.built_retriver(uploaded_files, chunk_size=200, chunk_overlap=20, k=5)
        
        # Option 2: Use MMR (Maximal Marginal Relevance) for diverse results
        _ = ci.built_retriver(
            uploaded_files, 
            chunk_size=200, 
            chunk_overlap=20, 
            k=5,                    # Final number of documents to return
            search_type="mmr",      # Use MMR instead of similarity
            fetch_k=20,             # Fetch 20 candidates before MMR filtering
            lambda_mult=0.5         # 0.5 = balance between relevance and diversity
        )

        # Close file handles
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_path=index_dir, k=5, index_name=os.getenv("FAISS_INDEX_NAME", "index"))

        # Interactive multi-turn chat loop
        chat_history = []
        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break

            answer = rag.invoke(user_input, chat_history=chat_history)
            print("Assistant:", answer)

            # Maintain conversation history for context in subsequent turns
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_document_ingestion_and_rag()
