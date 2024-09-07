#!/bin/bash
echo "[START] - run script"

echo "[START] - install langchain lib"
python -m pip install langchain openai langchain-openai langchain-community langchainhub sentence_transformers langchain-huggingface

echo "[START] - install env lib"
python -m pip install python-dotenv

echo "[START] - install chromadb lib"
python -m pip install chromadb

echo "[DONE] run script"