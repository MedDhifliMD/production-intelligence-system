#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
# UPDATED IMPORTS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


CONTEXT_FILE = "rag_context.pkl"
VECTORSTORE_FILE = "rag_vectorstore.faiss"

# -------------------------
# 1️⃣ Load full context
# -------------------------
print(" Loading context...")
with open(CONTEXT_FILE, "rb") as f:
    full_text = pickle.load(f)

# -------------------------
# 2️⃣ Split context into chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_text(full_text)
print(f" Context split into {len(chunks)} chunks.")

# -------------------------
# 3️⃣ Generate embeddings
# -------------------------
print(" Generating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # small, local-friendly

vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local(VECTORSTORE_FILE)
print(f" Embeddings saved to '{VECTORSTORE_FILE}'")