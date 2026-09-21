# Module 3: GenAI Support Assistant (RAG Service)

## 1. Pipeline Architecture Walkthrough
The RAG pipeline implements an end-to-end grounded customer support flow orchestrated through LangGraph:
- **Ingestion (`setup_docs.py`)**: Loads 8 raw Zepto operational policy files defining delivery thresholds, returns, memberships, cancellation, damaged goods, gift cards, and support availability.
- **Embedding & Storage (`ingest.py`)**: Generates 384-dimensional dense vector embeddings using the open-source sentence-transformers model `all-MiniLM-L6-v2` and persists them into a local ChromaDB collection (`zepto_policies`) with cosine similarity.
- **Intent Routing & Retrieval (`main.py`)**: A LangGraph `StateGraph` routes queries via a conditional edge (`classify_intent`). Queries containing policy keywords route to `retrieve_and_answer` where ChromaDB retrieves the top-3 nearest document chunks. General queries route directly to `direct_answer`.
- **Generation & Validation**: Under the default `MOCK_LLM=1` graded baseline, answers are deterministically formatted (`Based on the retrieved context: <snippet>`) without external API dependencies. Under `MOCK_LLM=0`, the pipeline passes retrieved context to an external LLM using a structured Role-Context-Task-Format-Length prompt skeleton with automatic Pydantic schema validation retries.

## 2. Default Baseline Test Call Transcripts (MOCK_LLM=1)

### Test Call 1: Policy Retrieval (`POST /ask`)
**Request:**
```json
{
  "query": "What is your delivery fee policy?"
}