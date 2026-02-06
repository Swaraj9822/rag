# Bedrock + Pinecone PDF RAG

This project provides a command-line RAG application that:

1. Reads a PDF.
2. Chunks text using structured boundaries (paragraphs/sections first, then sentence-aware splitting when needed).
3. Generates embeddings using **Amazon Titan Embeddings** (`amazon.titan-embed-text-v1`, 1536 dimensions).
4. Stores vectors in **Pinecone**.
5. Answers user questions using **Meta Llama 3 70B Instruct** on AWS Bedrock (`meta.llama3-3-70b-instruct-v1:0`).

## Prerequisites

- Python 3.10+
- AWS credentials with Bedrock access for:
  - `amazon.titan-embed-text-v1`
  - `meta.llama3-3-70b-instruct-v1:0`
- Pinecone API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` values:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=bedrock-rag-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Ingest a PDF

```bash
python rag_app.py ingest --pdf /path/to/document.pdf --namespace mydocs
```

## Ask a Question

```bash
python rag_app.py ask --question "What is the warranty period?" --namespace mydocs --top-k 4
```

The chunker is structured-first to preserve context (paragraph/section grouping) instead of relying only on fixed-size windows. The prompt is tuned to be concise and to-the-point, and to return a fallback response when the answer does not exist in retrieved context.
