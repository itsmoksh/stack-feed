import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
import os
from chonkie import SemanticChunker
from dotenv import load_dotenv
from uuid import uuid4
from groq import Groq
load_dotenv()
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
            print("Deleted previous weekly digest collection")
            #Creating a same collection again
            client.create_collection(
                collection_name="weekly_digest",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print("Created weekly digest collection")
        else: #if the bot is running for first time.
            client.create_collection(
                collection_name="weekly_digest",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    points = chunk_plus_points()
    # Ingesting points into the collection
    print("Ingesting points to collection")
    client.upsert(
        collection_name="weekly_digest",
        points=points,
    )

def chunk_plus_points():
    with open('latest_news.json','r') as f:
        latest_news = json.load(f)

    chunker = SemanticChunker(
        embedding_model="minishlab/potion-retrieval-32M",
        threshold=0.8,  # Similarity threshold (0-1)
        chunk_size=2048,  # Maximum tokens per chunk
        similarity_window=3,  # Window for similarity calculation
        min_sentences_per_chunk=6
    )
    print("Chunking Articles")
    points = []
    for category,articles in latest_news.items():
        for ar_id, article in enumerate(articles):
            chunk = chunker.chunk(article['content'])
            for chunk_id, chunk in enumerate(chunk):
                point = PointStruct(
                    id = str(uuid4()),
                    vector=Document(
                        text= chunk.text, model='sentence-transformers/all-MiniLM-L6-v2'
                    ),
                    payload={
                        'chunk_id':f'{category}_article_{ar_id}_chunk_{chunk_id}',
                        'text':chunk.text,
                        'title': article['title'],
                        'category': category,
                        'source': article['source'],
                    }
                )
                points.append(point)

    return points

def get_relevant_qa(query):
    results = client.query_points(
        collection_name="weekly_digest",
        query=Document(text=query, model="sentence-transformers/all-MiniLM-L6-v2"),
        with_payload=True,
        limit=2
    )
    context = ''.join([r.payload['text'] for r in results.points])
    prompt = f'''Given the question and context below, generate the answer based on context only,
        and answer as you are telling to the user.
        If you don't find the answer inside the context, say I don't know.
        Do not make up things.

        Question: {query}
        Context: {context}'''

    completion = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{"role": "user", "content": prompt}])

    return completion.choices[0].message.content


if __name__ == "__main__":
    # ingest_digest()
    print(get_relevant_qa("Is there any cost associated with GPT 5.5 model"))

