# Smart Travel Assistant - France Edition

> A conversational RAG (Retrieval-Augmented Generation) chatbot that acts as a specialist tour guide for famous landmarks in France, built with LangChain, Groq, and FAISS.
> 
> ![Smart Travel Assistant Demo](demo.gif)

## About The Project

This project is a fully functional conversational AI assistant built using the RAG (Retrieval-Augmented Generation) pattern. The goal was to create a chatbot that could act as a knowledgeable and reliable tour guide for a specific domain. In this case, famous landmarks in France, without the risk of AI hallucination.

The application ingests data from live web pages, processes it into a searchable vector knowledge base, and uses this knowledge base to provide grounded, context-aware answers to user questions. It is designed to be conversational, capable of handling follow-up questions and topic changes intelligently.

---

## Key Features

*   **Web-Aware Data Ingestion:** Automatically scrapes and processes text from a list of URLs (Wikipedia articles) to build its knowledge base.
*   **Conversational RAG Pipeline:** Implements an advanced RAG architecture:
    *   **Question Rewriting:** Uses an LLM to rephrase follow-up questions into self-contained queries, ensuring accurate context is maintained.
    *   **Retrieval:** Performs semantic search against a FAISS vector store to find the most relevant information.
    *   **Generation:** Uses the Groq API with Llama3 to generate answers that are strictly grounded in the retrieved context.
*   **Conversational Memory:**
    *   Maintains a short-term chat history to understand follow-up questions.
    *   Tracks used documents to avoid giving repetitive answers in a longer conversation.
*   **Hallucination Prevention:** The system prompt is engineered to force the LLM to rely solely on the provided documents, politely declining to answer if the information is not in its knowledge base.
*   **Source Transparency:** Displays the exact source text that was used to generate an answer, providing verifiability for the user.

## Tech Stack

*   **LLM & API:** Groq API (Llama3 8B)
*   **AI Framework:** LangChain
*   **Vector Store:** FAISS (Facebook AI Similarity Search)
*   **Embedding Model:** `all-MiniLM-L6-v2`
*   **Web Framework:** Flask
*   **Core Libraries:** `python-dotenv`, `beautifulsoup4`, `numpy`
