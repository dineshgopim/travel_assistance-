from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import FAISS 

# Webpages to load
urls = [
    "https://en.wikipedia.org/wiki/Eiffel_Tower",
    "https://en.wikipedia.org/wiki/Louvre",
    "https://en.wikipedia.org/wiki/Palace_of_Versailles",
    "https://en.wikipedia.org/wiki/Mont-Saint-Michel",
    "https://en.wikipedia.org/wiki/Notre-Dame_de_Paris"
]

print("Loading documents from webpages...", flush=True)
loader = WebBaseLoader(urls)
documents = loader.load()
print(f"Loaded {len(documents)} documents from the web.", flush=True)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
print(f"Split the content into {len(chunks)} chunks.", flush=True)

# Use OpenAI embeddings (requires OPENAI_API_KEY)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

print("Creating FAISS vector store and embeddings (this may take a while)...", flush=True)
vector_store = FAISS.from_documents(chunks, embedding_function)

# Save FAISS index
vector_store.save_local("faiss_index_travel")
print("FAISS vector store created and saved to 'faiss_index_travel' directory.", flush=True)
