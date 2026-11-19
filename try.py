import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "BAAI/bge-reranker-v2-m3"

# Automatically select GPU (CUDA or Apple Silicon) if available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# Optional: Use half-precision (fp16) on CUDA for faster inference
if device == "cuda":
    model.half()

query = "What is deep learning?"
documents = [
    "The weather is nice today.",
    "Deep learning is a subset of machine learning that uses multi-layered neural networks.",
    "I enjoy eating pizza on weekends.",
    "Neural networks allow computers to learn patterns from large amounts of data."
]

# Create [query, document] pairs
pairs = [[query, doc] for doc in documents]

with torch.no_grad():
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=8192  # bge-reranker-v2-m3 supports up to 8192 tokens
    ).to(device)

    # Get raw logits and apply sigmoid to normalize scores between 0 and 1
    logits = model(**inputs, return_dict=True).logits.view(-1,).float()
    scores = torch.sigmoid(logits).cpu().tolist()

# Sort documents from highest to lowest score
ranked_results = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

for score, doc in ranked_results:
    print(f"Score: {score:.4f} | Document: {doc}")
