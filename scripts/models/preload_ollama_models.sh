#!/bin/bash
# Preload common LLM models for offline use with Ollama

echo "Pulling tinyllama..."
ollama pull tinyllama

echo "Pulling llama3.2:1b..."
ollama pull llama3.2:1b

echo "Model preloading complete."
