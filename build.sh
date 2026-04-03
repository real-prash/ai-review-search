#!/bin/bash
# Pre-download the FastEmbed model during Railway's build step so the
# first query doesn't incur a 30-60s cold start download.
python -c "
from langchain_community.embeddings import FastEmbedEmbeddings
print('Pre-downloading embedding model...')
FastEmbedEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print('Model cached.')
"
