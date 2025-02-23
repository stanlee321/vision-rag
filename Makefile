build:
	docker buildx build --platform linux/amd64 -t stanlee321/rag-api:latest --load -f Dockerfile .

run:
	docker compose up -d

