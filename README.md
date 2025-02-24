# Litle RAG with Vision

## Overview

This is a simple RAG API built with FastAPI and LlamaIndex. It allows you to upload a PDF file and query it using a natural .
It uses Chroma as a vector database and OpenTelemetry for tracing.

It also uses Vision to extract text from images using `smart_llm_loader`

## Setup

```bash
pip install -r requirements.txt

docker compose up -d # for chroma
```

## Test CURL

```bash
curl -X POST "http://localhost:8003/v1/rag/upload?collection_name=test_collection_smart&loader=smart" \
     -F "file=@./data/2502.06472v1.pdf"

curl -X POST "http://23.20.190.185:8003/v1/rag/upload?collection_name=test_collection_low&loader=low" \
     -F "file=@./data/mexico.pdf" \
     -H "Authorization: Bearer tP07DAahaFF\!"
     

curl -X POST "http://localhost:8003/v1/rag/upload?collection_name=test_collection_low&loader=low" \
     -F "file=@./data/2502.06472v1.pdf"

# with auth
curl -X POST "http://23.20.190.185:8003/v1/rag/upload?collection_name=test_collection_low&loader=low" \
     -F "file=@./data/test.pdf" \
     -H "Authorization: Bearer tP07DAahaFF\!"


curl -G "http://localhost:8003/v1/rag/query" \
     --data-urlencode "q=What is the document about?" \
     --data-urlencode "collection_name=test_collection_smart" \
     --data-urlencode "response_mode=compact"

curl -G "http://localhost:8003/v1/rag/query" \
     --data-urlencode "q=What is the document about?" \
     --data-urlencode "collection_name=test_collection_low" \
     --data-urlencode "response_mode=tree_summarize"


curl "http://23.20.190.185:8003/v1/rag/query?q=What+is+the+main+idea+of+the+document%3F&collection_name=governingDocuments&response_mode=compact&doc_type=governingDocuments"

curl "http://23.20.190.185:8003/v1/rag/collections" -H "Authorization: Bearer tP07DAahaFF\!"

curl "http://localhost:8003/v1/rag/info"

curl -H "Authorization: Bearer tP07DAahaFF\!" \
     "http://23.20.190.185:8003/v1/rag/query?q=acerca+del+articulo+50+de+la+constitucion+de+mexico%3F&collection_name=default_collection&response_mode=compact"

curl "http://23.20.190.185:8003/v1/rag/collections" -H "Authorization: Bearer tP07DAahaFF\!"

```



