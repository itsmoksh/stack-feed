from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models
from groq import Groq
from tqdm import tqdm
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from chonkie import SemanticChunker
from logging_setup import setup_logger
load_dotenv()

#Creating a Qdrant Client
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    cloud_inference=True,
    timeout=60
)
#Groq Client
groq_client = Groq()

news_path = Path(__file__).parent.parent/'latest_news.json'
dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sparse_model_name = "qdrant/bm25"

#Setting up the rag logger
log_path = Path(__file__).parent.parent/'stack_feed.log'
rag_logger = setup_logger('rag_log',log_path)

def setup_collection():
    client.create_collection(
        collection_name="weekly_digest",
        vectors_config={
            'dense': models.VectorParams(
                size=client.get_embedding_size(dense_model_name),
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={'sparse': models.SparseVectorParams()},
    )

def ingest_digest(refresh = True):
    if refresh:
        if client.collection_exists("weekly_digest"):
            client.delete_collection(collection_name="weekly_digest")
            rag_logger.info("Deleted the 'weekly_digest' collection ")

            # Creating a same collection again
        setup_collection()
        rag_logger.info("Recreated the 'weekly_digest' collection")

        docs, metadata = chunker()
        rag_logger.info("Chunks created, storing them into the Qdrant database")
        client.upload_collection(
            collection_name="weekly_digest",
            vectors=tqdm(docs),
            payload=metadata,
            parallel=4,
        )
        rag_logger.info("Chunks uploaded, and ready for QnA..")

    else:
        rag_logger.info("Refresh is False, No need to create a new collection.")
        return

def chunker():
    with open(news_path, 'r') as f:
        latest_news = json.load(f)
    rag_logger.info("Loaded latest news from the json file, and creating a chunker")

    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        threshold=0.8,  # Similarity threshold (0-1)
        chunk_size=2048,  # Maximum tokens per chunk
        similarity_window=3,  # Window for similarity calculation
        min_sentences_per_chunk=4
    )
    rag_logger.info("Chunker created, chunking the latest news and embedding it using 'all-MiniLM-L6-v2'")
    docs = []
    metadata = []
    for category,articles in latest_news.items():
        for ar_id, article in enumerate(articles):
            chunk = chunker.chunk(article['content'])
            for chunk_id, chunk in enumerate(chunk):
                dense_document = models.Document(text=chunk.text, model=dense_model_name)
                sparse_document = models.Document(text=chunk.text, model=sparse_model_name)
                docs.append(
                    {
                        'dense': dense_document,
                        'sparse': sparse_document,
                    })
                metadata.append({
                    'chunk_id': f'{category}_article_{ar_id}_chunk_{chunk_id}',
                    'text': chunk.text,
                    'title': article['title'],
                    'category': category,
                    'source': article['url'],
                })
    return docs, metadata

def retrieve_chunks(query: str):
    search_result = client.query_points(
        collection_name='weekly_digest',
        query=models.FusionQuery(
            fusion=models.Fusion.RRF  ),
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=dense_model_name),
                using='dense',
            ),
            models.Prefetch(
                query=models.Document(text=query, model=sparse_model_name),
                using='sparse',
            ),
        ],
        query_filter=None,
        limit=3
    ).points

    chunk_ids = ",".join([r.payload['chunk_id'] for r in search_result])
    chunk_sources = set([r.payload['source'] for r in search_result])
    sources = ", ".join([source for source in chunk_sources])
    context = ''.join([r.payload['text'] for r in search_result])
    search_logs= {
        'Query':query,
        'chunk_ids':chunk_ids,
        'source':sources
    }
    rag_logger.info(search_logs)
    return chunk_ids, sources, context

def generate_response(query,context):
    prompt = f'''
        You are an AI news assistant.
    
    The provided context is the ONLY source of truth.
    
    RULES:
    1. Answer ONLY using the provided context.
    2. Do NOT use external knowledge.
    3. Do NOT infer or assume missing information.
    4. If the answer is not present in the context,
       respond Exactly with:
       This is not covered in the provided AI news articles. Also explain what is covered in the given context and what not which is required for answering the question
    
    COMPARISON RULES:
    1. Only compare items if sufficient information exists for ALL items.
    2. If information for any item is missing:
       respond EXACTLY with:
       "Cannot compare between <item1> and <item2> and so on, because sufficient information is not available about <item> in the provided articles."
    3. Do NOT provide partial comparisons.
    4. Do NOT make assumptions or rankings.
    
    RESPONSE FORMAT:
    - Provide a direct answer.
    - Use concise structured formatting.
    - Mention only information explicitly supported by context.
    
    Question:
    {query}
    
    Context:
    {context}
'''

    completion = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{"role": "user", "content": prompt}],
        temperature=0)

    return completion

if __name__ == '__main__':
    ingest_digest(refresh=False)
    query = 'What are the safety mechanism introduced in Sol'
    _,sources,context = retrieve_chunks(query)
    answer= generate_response(query,context)
    print(answer.choices[0].message.content)
    print(sources)
