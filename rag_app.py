import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from text_utils import split_text

import boto3
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader


EMBED_MODEL_ID = "amazon.titan-embed-text-v1"
LLM_MODEL_ID = "meta.llama3-3-70b-instruct-v1:0"
EMBED_DIMENSION = 1536


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int


class BedrockClient:
    def __init__(self, region: str) -> None:
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def embed(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId=EMBED_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    def generate(self, prompt: str, max_tokens: int = 350) -> str:
        request_body = {
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_gen_len": max_tokens,
        }
        response = self.client.invoke_model(
            modelId=LLM_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )
        payload = json.loads(response["body"].read())
        return payload.get("generation", "").strip()


class PineconeStore:
    def __init__(self, api_key: str, index_name: str, cloud: str, region: str) -> None:
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self._ensure_index(cloud, region)
        self.index = self.pc.Index(index_name)

    def _list_index_names(self) -> List[str]:
        raw = self.pc.list_indexes()
        if hasattr(raw, "names"):
            return list(raw.names())
        if isinstance(raw, list):
            return [getattr(item, "name", item.get("name")) for item in raw]
        if isinstance(raw, dict):
            indexes = raw.get("indexes", raw)
            return [item["name"] if isinstance(item, dict) else item.name for item in indexes]
        return []

    def _ensure_index(self, cloud: str, region: str) -> None:
        if self.index_name in self._list_index_names():
            return
        self.pc.create_index(
            name=self.index_name,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    def upsert(self, vectors: Iterable[dict], namespace: str) -> None:
        self.index.upsert(vectors=list(vectors), namespace=namespace)

    def query(self, vector: List[float], namespace: str, top_k: int = 4):
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )



def load_pdf_chunks(pdf_path: Path) -> List[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks: List[Chunk] = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for idx, chunk_text in enumerate(split_text(page_text), start=1):
            digest = hashlib.sha1(f"{pdf_path}-{page_num}-{idx}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=digest,
                    text=chunk_text,
                    source=pdf_path.name,
                    page=page_num,
                )
            )
    return chunks


def format_prompt(context_chunks: List[dict], question: str) -> str:
    context = "\n\n".join(
        [
            f"[Source: {item['metadata']['source']} | Page: {item['metadata']['page']}]\n{item['metadata']['text']}"
            for item in context_chunks
        ]
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a document question-answering assistant. Answer precisely and to the point. "
        "Use only facts from the provided context. If the answer is not present, say: "
        "'I could not find that in the provided document.'\n"
        "Cite source page numbers in parentheses, e.g. (page 3).<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Context:\n{context}\n\nQuestion: {question}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


def ingest(pdf_path: str, namespace: str) -> None:
    load_dotenv()
    region = os.environ["AWS_REGION"]
    pinecone_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ.get("PINECONE_INDEX_NAME", "bedrock-rag-index")
    pinecone_cloud = os.environ.get("PINECONE_CLOUD", "aws")
    pinecone_region = os.environ.get("PINECONE_REGION", region)

    bedrock = BedrockClient(region=region)
    store = PineconeStore(
        api_key=pinecone_key,
        index_name=index_name,
        cloud=pinecone_cloud,
        region=pinecone_region,
    )

    chunks = load_pdf_chunks(Path(pdf_path))
    vectors = []
    for chunk in chunks:
        embedding = bedrock.embed(chunk.text)
        vectors.append(
            {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                },
            }
        )

    store.upsert(vectors=vectors, namespace=namespace)
    print(f"Ingested {len(vectors)} chunks from {pdf_path} into namespace '{namespace}'.")


def ask(question: str, namespace: str, top_k: int) -> None:
    load_dotenv()
    region = os.environ["AWS_REGION"]
    pinecone_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ.get("PINECONE_INDEX_NAME", "bedrock-rag-index")
    pinecone_cloud = os.environ.get("PINECONE_CLOUD", "aws")
    pinecone_region = os.environ.get("PINECONE_REGION", region)

    bedrock = BedrockClient(region=region)
    store = PineconeStore(
        api_key=pinecone_key,
        index_name=index_name,
        cloud=pinecone_cloud,
        region=pinecone_region,
    )

    query_vector = bedrock.embed(question)
    matches = store.query(vector=query_vector, namespace=namespace, top_k=top_k)
    context_chunks = matches.get("matches", [])

    prompt = format_prompt(context_chunks=context_chunks, question=question)
    answer = bedrock.generate(prompt)

    print("\nAnswer:\n")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG with AWS Bedrock + Pinecone")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into Pinecone")
    ingest_parser.add_argument("--pdf", required=True, help="Path to PDF document")
    ingest_parser.add_argument(
        "--namespace",
        default="default",
        help="Pinecone namespace to store chunks",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question from ingested docs")
    ask_parser.add_argument("--question", required=True, help="Question to ask")
    ask_parser.add_argument(
        "--namespace",
        default="default",
        help="Pinecone namespace to query",
    )
    ask_parser.add_argument("--top-k", type=int, default=4, help="Top chunks to retrieve")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(pdf_path=args.pdf, namespace=args.namespace)
    elif args.command == "ask":
        ask(question=args.question, namespace=args.namespace, top_k=args.top_k)


if __name__ == "__main__":
    main()
