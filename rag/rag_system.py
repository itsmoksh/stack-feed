import json
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import os
from chonkie import SemanticChunker
from dotenv import load_dotenv
from uuid import uuid4
from groq import Groq
from fastembed import TextEmbedding
from pathlib import Path
from logging_setup import setup_logger
load_dotenv()

#Setting up the rag logger
log_path = Path(__file__).parent.parent/'stack_feed.log'
rag_logger = setup_logger('rag_log',log_path)

embed_model = TextEmbedding('nomic-ai/nomic-embed-text-v1.5')
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# connect to Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True
)
groq_client = Groq()

def ingest_digest(refresh = True):
    if refresh:
        if client.collection_exists("weekly_digest"):
            client.delete_collection(collection_name="weekly_digest")
            rag_logger.info("Deleted the 'weekly_digest' collection ")

            #Creating a same collection again
            client.create_collection(
                collection_name="weekly_digest",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            rag_logger.info("Recreated the 'weekly_digest' collection")

        else: #if the bot is running for first time.
            client.create_collection(
                collection_name="weekly_digest",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
    docs,metadata = chunker()
    rag_logger.info("Chunks created, storing them into the Qdrant database")
    client.upsert(
        collection_name="weekly_digest",
        points=models.Batch(
            ids=[str(uuid4()) for i in range(len(docs))],
            vectors=[next(embeddings).tolist() for embeddings in docs],
            payloads=metadata,
        ),
    )
    rag_logger.info("Chunks upserted, and ready for QnA..")

def chunker():
    with open('../latest_news.json', 'r') as f:
        latest_news = json.load(f)
    rag_logger.info("Loaded latest news from the json file, and creating a chunker")

    chunker = SemanticChunker(
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        threshold=0.8,  # Similarity threshold (0-1)
        chunk_size=2048,  # Maximum tokens per chunk
        similarity_window=3,  # Window for similarity calculation
        min_sentences_per_chunk=4
    )
    rag_logger.info("Chunker created, chunking the latest news and embedding it using 'nomic-ai/nomic-embed-text-v1.5'")
    docs = []
    metadata = []
    for category,articles in latest_news.items():
        for ar_id, article in enumerate(articles):
            chunk = chunker.chunk(article['content'])
            for chunk_id, chunk in enumerate(chunk):
                docs.append(embed_model.embed([chunk.text]))
                metadata.append({
                    'chunk_id': f'{category}_article_{ar_id}_chunk_{chunk_id}',
                    'text': chunk.text,
                    'title': article['title'],
                    'category': category,
                    'source': article['source'],
                })
    return docs, metadata

def get_relevant_qa(query):
    rag_logger.info(f"User Query: {query}, finding the relevant answers")
    query_embed = next(embed_model.embed([query]))
    results = client.query_points(
        collection_name="weekly_digest",
        query=query_embed.tolist(),
        with_payload=True,
        limit=2
    )
    rag_logger.info(f"Relevant chunks:{','.join([r.payload['chunk_id']for r in results.points])}")
    context = ''.join([r.payload['text'] for r in results.points])
    prompt = f'''
    You are an expert in understanding the context of the articles about the advancements in AI or updates, and
    provided with the user question and the context.
    Based on the context only you have to generate the answer.
    If you don't find the answer inside the context, return their is no context provided in the above news.
    Do not make up things.
    Question: {query}
    Context: {context}'''

    completion = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{"role": "user", "content": prompt}],
        temperature=0)

    return completion.choices[0].message.content


if __name__ == "__main__":
    ingest_digest()
    print(get_relevant_qa("Is there any cost associated with GPT 5.5 model"))