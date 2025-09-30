# MMR (Maximal Marginal Relevance) Implementation Guide

## What is MMR?

**Maximal Marginal Relevance (MMR)** is a retrieval algorithm that balances relevance and diversity in search results. Instead of just returning the most similar documents, MMR ensures variety by reducing redundancy.

### How MMR Works:

1. **Initial Retrieval**: Fetch `fetch_k` documents using similarity search
2. **Re-ranking**: Select `k` documents that maximize:
   - **Relevance** to the query
   - **Diversity** from already selected documents

### Key Parameters:

- **`k`**: Number of final documents to return (e.g., 5)
- **`fetch_k`**: Number of documents to initially fetch before re-ranking (e.g., 20)
  - Should be larger than `k` (typically 2-4x)
- **`lambda_mult`**: Diversity parameter (range: 0 to 1)
  - `0.0` = Maximum diversity (least similar documents)
  - `1.0` = Maximum relevance (ignores diversity)
  - `0.5` = Balanced approach (recommended default)

---

## Implementation in Your Project

### 1. Configuration (`config.yaml`)

```yaml
retriever:
  top_k: 10
  search_type: "mmr"  # Options: "similarity", "mmr", "similarity_score_threshold"
  fetch_k: 20        # Number of docs to fetch before MMR re-ranking
  lambda_mult: 0.5   # Diversity vs relevance balance
```

### 2. Document Ingestion (`data_ingestion.py`)

```python
# Build retriever with MMR
retriever = ci.built_retriver(
    uploaded_files, 
    chunk_size=200, 
    chunk_overlap=20, 
    k=5,
    search_type="mmr",
    fetch_k=20,
    lambda_mult=0.5
)
```

### 3. Retrieval (`retrieval.py`)

```python
# Load retriever from FAISS with MMR
rag.load_retriever_from_faiss(
    index_path=index_dir, 
    k=5, 
    search_type="mmr",
    fetch_k=20,
    lambda_mult=0.5
)
```

### 4. FastAPI Application (`main.py`)

Both upload and chat endpoints now use MMR by default:

```python
# Upload endpoint
ingestor.built_retriver(
    uploaded_files=wrapped_files,
    search_type="mmr",
    fetch_k=20,
    lambda_mult=0.5
)

# Chat endpoint
rag.load_retriever_from_faiss(
    index_path=index_path,
    search_type="mmr",
    fetch_k=20,
    lambda_mult=0.5
)
```

---

## When to Use MMR vs Similarity Search

### Use MMR When:
✅ You need diverse perspectives in answers  
✅ Documents contain repetitive information  
✅ You want to avoid redundant chunks  
✅ Building conversational AI with varied context  
✅ Documents have overlapping content  

### Use Similarity Search When:
✅ You need the most relevant results only  
✅ Precision is more important than diversity  
✅ Documents are already diverse  
✅ You want faster retrieval (no re-ranking)  

---

## Tuning MMR Parameters

### `fetch_k` (Initial Retrieval Size)
- **Low (10-15)**: Faster, less diversity
- **Medium (20-30)**: Balanced (recommended)
- **High (40+)**: More diversity options, slower

### `lambda_mult` (Diversity Control)
- **0.0-0.3**: High diversity, may sacrifice relevance
- **0.4-0.6**: Balanced (recommended starting point)
- **0.7-1.0**: Prioritizes relevance over diversity

---

## Testing MMR

Run the test script to see MMR in action:

```bash
python test.py
```

To compare with similarity search, modify `test.py`:

```python
# Switch to similarity search
retriever = ci.built_retriver(
    uploaded_files, 
    chunk_size=200, 
    chunk_overlap=20, 
    k=5,
    search_type="similarity"  # Changed from "mmr"
)
```

---

## Performance Considerations

- **MMR is slightly slower** than pure similarity search due to re-ranking
- **Trade-off**: Speed vs. result diversity
- For most conversational AI applications, the diversity benefits outweigh the minimal performance cost

---

## Modified Files

1. ✅ `multi_doc_chat/config/config.yaml` - Added MMR configuration
2. ✅ `multi_doc_chat/src/document_ingestion/data_ingestion.py` - Added MMR parameters
3. ✅ `multi_doc_chat/src/document_chat/retrieval.py` - Added MMR support in loader
4. ✅ `main.py` - Updated API endpoints to use MMR
5. ✅ `test.py` - Updated test script with MMR examples

---

## Example: Impact of MMR

### Query: "What is attention mechanism?"

**Similarity Search** might return:
1. Attention mechanism definition
2. Another attention mechanism definition (redundant)
3. Attention mechanism benefits (similar)
4. More about attention mechanism (similar)
5. Attention formula variation (similar)

**MMR Search** returns:
1. Attention mechanism definition
2. Attention mechanism benefits
3. Comparison with other mechanisms
4. Use cases in transformers
5. Historical context of attention

MMR provides a more comprehensive understanding by ensuring diverse information!

