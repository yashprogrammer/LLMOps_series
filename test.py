import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


def test_document_ingestion_and_rag():
    try:
        test_files = [
            "/Users/yashpatil/Developer/AI/YT/Sunny/LLMOps_series/data/NIPS-2017-attention-is-all-you-need-Paper.pdf",
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
        
        # Using MMR (Maximal Marginal Relevance) for diverse results
        # MMR parameters:
        # - fetch_k: Number of documents to fetch before MMR re-ranking (20)
        # - lambda_mult: Diversity parameter (0=max diversity, 1=max relevance, 0.5=balanced)
        retriever = ci.built_retriver(
            uploaded_files, 
            chunk_size=200, 
            chunk_overlap=20, 
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )
        
        # Alternative: Use similarity search instead of MMR
        # retriever = ci.built_retriver(uploaded_files, chunk_size=200, chunk_overlap=20, k=5, search_type="similarity")

        # Close file handles
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        # Load RAG with MMR search
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir, 
            k=5, 
            index_name=os.getenv("FAISS_INDEX_NAME", "index"),
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

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