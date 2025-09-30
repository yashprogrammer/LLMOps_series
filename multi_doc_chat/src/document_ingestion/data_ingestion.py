from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
import json
import uuid
from datetime import datetime
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
import hashlib
import sys


def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


class ChatIngestor:
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e


    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"

    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks

    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            ## FAISS manager very very important class for the docchat
            fm = FaissManager(self.faiss_dir, self.model_loader)

            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)

            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            # Configure search parameters based on search type
            search_kwargs = {"k": k}
            
            if search_type == "mmr":
                # MMR needs fetch_k (docs to fetch) and lambda_mult (diversity parameter)
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR search", k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
            
            return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e



SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# FAISS Manager (load-or-create)
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}} ## this is dict of rows

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists


        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self)-> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")


    def add_documents(self,docs: List[Document]):

        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")

        new_docs: List[Document] = []

        for d in docs:

            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs


        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs