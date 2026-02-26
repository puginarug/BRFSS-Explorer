# ScreenMind — CLAUDE.md

## Project context
ScreenMind is a learning project by David Zingerman, a computational biologist
(M.Sc., Weizmann Institute) transitioning to industry data science.

This is an end-to-end intelligent mental health risk assessment system that
combines a PyTorch MLP with a LangGraph agentic layer, RAG over scientific
papers, and a FastAPI deployment in Docker.

## How to work with me
- Explain what you're doing and WHY at each step
- Don't just generate code — teach me the concepts
- Follow the milestone structure; STOP and wait for me between milestones
- Prefer clarity over cleverness in code
- If I say "I don't understand X", stop and explain it differently before continuing
- If I say "just do it", you can skip explanations for that step only

## Tech stack
- Python 3.11+, PyTorch (model and training — no sklearn for modeling)
- scikit-learn (preprocessing only: StandardScaler, train_test_split)
- LangGraph + LangChain (agent orchestration)
- Anthropic Claude API (explanation synthesis)
- ChromaDB (vector store for RAG)
- Weights & Biases (experiment tracking)
- FastAPI + Uvicorn (serving)
- Pydantic (structured outputs + validation)
- Docker (containerization)

## Project goal
Predict mental health risk from BRFSS (CDC) survey data.
Wrap the model in a LangGraph agent that retrieves scientific context (RAG)
and returns structured, biologically-grounded explanations via the Claude API.
Deploy as a FastAPI app in Docker with a live public URL.

## Milestones
1. Project setup (this milestone — folder structure, deps, config)
2. Data acquisition & EDA (BRFSS 2023 download, feature exploration)
3. Feature engineering & preprocessing (task definition, splits, normalization)
4. PyTorch model (Dataset, DataLoader, MLP, training loop from scratch)
5. Experiment tracking with W&B (3 model variants, dashboard comparison)
6. RAG knowledge base (ChromaDB, embeddings, scientific paper retriever)
7. LangGraph agentic layer (3-node graph, structured JSON output)
8. FastAPI + Docker + deployment (containerize, deploy, live URL)
9. Polish & portfolio presentation (README, docstrings, hiring-manager story)
