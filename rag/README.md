# RAG Component - Stack Feed

The RAG (Retrieval Augmented Generation) Pipeline is an intelligent Q&A system of Stack Feed designed to ask questions about the weekly AI news digest and receive accurate, grounded answers using a hybrid semantic search and a Large language model.

### How it works:
- Chunk the article contents using Chonkie based on semantic similarity.
- Ingest these chunks into the Qdrant database as a dense embedding(all-MiniLM-L6-v2) and sparse embedding(BM25). 
- Find the top 4 article chunks based on the similarity with user query(hybrid search). 
- Generate structure and human-readable output using Grok llama-3.3-70b-versatile model. 
- Continuously evaluate certain metrics(LLM as a Judge), such as:
  - Ground-ness
  - context-relevant score
  - answer-relevant score
  - latency
  - fallback queries and low-score queries. 

## Architecture
![RAG Architecture](../assets/rag_workflow.png)

## RAG Requirements:
You need to create a cluster on Qdrant vector database
- Create a API and URL for that cluster 
- To prevent limit, also create a Hugging face token
- Store it in a .env file present in project root
```text
    QDRANT_URL=your_qdrant_cloud_url
    QDRANT_API_KEY=your_qdrant_api_key
    HF_TOKEN=your_huggingface_token
    
```
