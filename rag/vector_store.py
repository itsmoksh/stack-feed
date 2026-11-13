from qdrant_client import QdrantClient
from qdrant_client.models import models
import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from logging_setup import setup_logger
from rag.chunking import StructureAwareChunker
import json
from tqdm import tqdm

load_dotenv()
filtered_articles_path = Path(__file__).resolve().parent.parent / "filtered_articles.json"
dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sparse_model_name = "qdrant/bm25"
collection_name = "weekly_digest_structured"

rag_logger = setup_logger('structured_rag_log')

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True,
    timeout=60,
)
groq_client = Groq()

def setup_collection() -> None:
    """Create the collection used by this standalone pipeline."""
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=client.get_embedding_size(dense_model_name),
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

def chunker(max_section_chars: int = 1048):
    """Load the latest categorized articles and prepare indexed vectors."""
    with filtered_articles_path.open("r", encoding="utf-8") as article_file:
        articles_by_category = json.load(article_file)

    structure_chunker = StructureAwareChunker(max_section_chars=max_section_chars)

    docs = []
    payloads = []

    for category, articles in articles_by_category.items():
        for ar_id, article in enumerate(articles):
            print("="*30, article["title"], "="*30)
            article_chunks = structure_chunker.chunk(
                article_text=article["content"],
                article_title=article["title"],
                category=category,
            )

            for chunk in article_chunks:
                print(chunk.embed_text)
                # Both retrieval channels see chunk.embed_text, which already
                # carries the [category] title — header: content prefix.
                docs.append(
                    {
                        "dense": models.Document(text=chunk.embed_text, model=dense_model_name),
                        "sparse": models.Document(text=chunk.embed_text, model=sparse_model_name),
                    }
                )

                payloads.append(
                    {
                        "chunk_id": f"{category}_article_{ar_id}_chunk_{chunk.chunk_index}",
                        "title": chunk.article_title,
                        "section": chunk.header,
                        "category": chunk.category,
                        "source": article["url"],
                        "text": (
                            f"Article: {chunk.article_title}\n"
                            + (f"Section: {chunk.header}\n" if chunk.header else "")
                            + chunk.text
                        ),
                    }
                )
                print(payloads[-1])

    return docs, payloads

def ingest_digest(refresh: bool = True) -> None:
    """Rebuild the standalone collection from filtered_articles.json."""
    if not refresh:
        rag_logger.info("Refresh is False; structured collection was not rebuilt.")
        return

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
        rag_logger.info("Deleted collection %s", collection_name)
    setup_collection()

    docs, payloads = chunker()
    rag_logger.info("Created %d structure-aware chunks", len(docs))
    if docs:
        client.upload_collection(
            collection_name=collection_name,
            vectors=tqdm(docs),
            payload=payloads,
            parallel=4,
        )
    rag_logger.info("Structured chunks uploaded and ready for Q&A")

if __name__ == "__main__":
    chunker()


