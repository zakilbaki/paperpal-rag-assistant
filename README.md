# PaperPal

RAG assistant for scientific PDFs with a FastAPI backend, Streamlit frontend, MongoDB storage, and Dockerized local deployment.

## Problem

Reading long technical papers is slow, and plain LLM chat is unreliable on dense PDFs. PaperPal extracts content from uploaded papers, stores structured information, and answers questions through retrieval rather than generic generation.

## Input Source

- user-uploaded PDF papers
- extracted text chunks and metadata
- MongoDB collections for papers, chunks, and embeddings

## Method

- PDF upload and parsing
- chunking and metadata extraction
- embedding-based retrieval
- API-first backend with FastAPI
- Streamlit UI for upload and interaction
- Docker and Render deployment configuration

## Current Project Scope

- PDF upload endpoint
- summarization flow
- keyword extraction flow
- comparison-ready backend structure
- containerized frontend/backend services

## Results to Surface

This repo already shows strong engineering scope, but it still needs clearer proof points. Add these before pinning:

- supported document size and upload flow
- example response quality on one sample paper
- latency for upload and summary generation
- one screenshot of the UI
- one screenshot of the API docs

## Project Structure

```text
paperpal-2/
├── backend/
│   ├── app/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
├── render.yaml
└── .env.example
```

## How To Run

```bash
cp .env.example .env
docker-compose up --build
```

Frontend: `http://localhost:8501`  
Backend docs: `http://localhost:8000/docs`

## Tech Stack

- Python
- FastAPI
- Streamlit
- MongoDB
- Docker
- Render


