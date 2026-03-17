#!/usr/bin/env python3
"""Launch the chat web server with Ollama RAG enabled. Requires Ollama running with nomic-embed-text."""
import runpy
import sys

if __name__ == "__main__":
    sys.argv = ["", "--rag"] + [
        a for a in sys.argv[1:] if "chat_ollama_rag" not in a
    ]
    runpy.run_module("scripts.chat_web", run_name="__main__")
