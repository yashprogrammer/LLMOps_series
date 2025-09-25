# MultiDocChat (FastAPI)

## How it works
- Upload: Files are uploaded to `data/<session_id>/`, split, embedded, and saved as a FAISS index in `faiss_index/<session_id>/`.
- Chat: Each request loads the FAISS index for the given `session_id` and answers using RAG.
- Sessions: A simple in-memory history per session on the server (resets on restart). The browser stores `session_id` in `localStorage`.

## Run locally
1. Install deps
```bash
pip install -r requirements.txt
```
2. Start the server
```bash
uvicorn main:app --reload
```
3. Open the UI
```bash
open http://localhost:8000/
```

## Endpoints
- `GET /` – Serves the UI.
- `GET /health` – Health check.
- `POST /upload` – Form-data file upload. Returns `{ session_id, indexed }`.
- `POST /chat` – JSON body `{ session_id, message }`. Returns `{ answer }`.

## Notes
- Ensure your API keys/config are set for the `ModelLoader` to load embeddings/LLM.
- Supported file types: `.pdf`, `.docx`, `.txt`.
- For production, add persistence for chat history and auth; consider cleanup of old session directories.

