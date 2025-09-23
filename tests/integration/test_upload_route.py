import io
import pytest


def test_upload_success_returns_session_and_indexed(client, clear_sessions, stub_ingestor, tmp_dirs):
    files = {"files": ("note.txt", io.BytesIO(b"hello world"), "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["indexed"] is True
    assert data["session_id"]


def test_upload_no_files_validation_error(client, clear_sessions, stub_ingestor):
    # Without files FastAPI validation will yield 422; send empty list to hit our 400
    resp = client.post("/upload", files=[])
    assert resp.status_code == 422


def test_upload_ingestor_failure_returns_500(client, clear_sessions, monkeypatch, tmp_dirs):
    import multi_doc_chat.src.document_ingestion.data_ingestion as di
    import main

    class Boom:
        def __init__(self, *a, **k):
            self.session_id = "sess_test"
        def built_retriver(self, *a, **k):
            from multi_doc_chat.exception.custom_exception import DocumentPortalException
            raise DocumentPortalException("boom", None)

    monkeypatch.setattr(di, "ChatIngestor", Boom)
    monkeypatch.setattr(main, "ChatIngestor", Boom)
    files = {"files": ("note.txt", io.BytesIO(b"hello world"), "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"].lower()