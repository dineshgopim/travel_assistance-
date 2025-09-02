import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from groq import Groq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
load_dotenv()

# -----------------------------
# Groq Client Setup
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Embeddings + FAISS Setup
# -----------------------------
print("Loading FAISS vector store...")
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("GROQ_API_KEY")  # ✅ uses Groq key
)

vector_store = FAISS.load_local(
    "faiss_index_travel",
    embeddings=embedding_function,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={'k': 10})
print("Vector store loaded successfully.")

# -----------------------------
# Global Variables
# -----------------------------
chat_history = []
used_docs = set()

# -----------------------------
# Utility Filters
# -----------------------------
@app.template_filter('nl2br')
def nl2br(value):
    return value.replace("\n", "<br>") if value else value

# -----------------------------
# Helper Functions
# -----------------------------
def create_standalone_question(question, history):
    """Rewrites follow-up questions into standalone ones."""
    if not history:
        return question

    history_str = "".join([f"{a.capitalize()}: {t}\n" for a, t in history])
    system_prompt = """
    Given a chat history and a follow-up question, rephrase the follow-up question 
    to be a standalone question that can be understood without the chat history.
    Do NOT answer the question, just reformulate it.
    """
    user_prompt = f"Chat History:\n---\n{history_str}\n---\nFollow Up Input: {question}"

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error rewriting question: {e}")
        return question


def generate_rag_answer(question, context):
    """Generates an answer based on retrieved docs."""
    system_prompt = """
    You are a specialist travel assistant for France. Your name is 'TourBot'.
    You MUST answer the user's question based ONLY on the provided context.
    - If the context contains the answer, provide it in a friendly and enthusiastic tone.
    - If the context does NOT contain the answer, you MUST say 
      "I'm sorry, I don't have information on that specific topic based on my current knowledge base."
    - DO NOT use any outside knowledge. DO NOT add any information that is not in the context.
    - Always refer to yourself as TourBot.
    """

    messages = [{"role": "system", "content": system_prompt}]
    for author, text in chat_history:
        messages.append({"role": author, "content": text})

    user_prompt = f"CONTEXT:\n{context}\n\nUSER'S QUESTION:\n{question}"
    messages.append({"role": "user", "content": user_prompt})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.0
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer."


# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['POST', 'GET'])
def index():
    global chat_history, used_docs
    last_question = None
    last_context = None

    if request.method == "POST":
        user_question = request.form['question'].strip()

        if user_question:
            # Step 1: Rewrite standalone question
            standalone_question = create_standalone_question(user_question, chat_history)
            print(f"--- STANDALONE QUESTION FOR RETRIEVER ---\n{standalone_question}\n-----------------------------------------")

            # Step 2: Retrieve relevant docs
            all_relevant_docs = retriever.invoke(standalone_question)
            new_docs = [doc for doc in all_relevant_docs if doc.page_content not in used_docs]
            final_docs = new_docs[:4]
            context = "\n\n".join([doc.page_content for doc in final_docs])

            # Step 3: Generate answer
            answer = generate_rag_answer(user_question, context)

            # Step 4: Update history
            chat_history.append(("user", user_question))
            chat_history.append(("assistant", answer))

            for doc in final_docs:
                used_docs.add(doc.page_content)

            if len(chat_history) > 8:
                chat_history = chat_history[-8:]

            last_question = user_question
            last_context = context

    return render_template("index.html", chat_history=chat_history, last_question=last_question, last_context=last_context)


@app.route('/clear', methods=['POST'])
def clear_history():
    global chat_history, used_docs
    chat_history = []
    used_docs = set()
    return render_template("index.html", chat_history=chat_history, last_question=None, last_context=None)


# -----------------------------
# Run Server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
