import os
import time
import shutil
import warnings

from extract import extract_text_from_multiple_pdfs
from embedding import create_vector_store_multi_doc
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama

warnings.filterwarnings("ignore")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"}
)

# 🧠 Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def run_chat():
    try:
        vectorstore = FAISS.load_local(
            "vector_store",
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    model = ChatOllama(model="tinyllama", temperature=0.0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=False  # switched off for now
    )
    qa_chain.output_key = "answer"

    print("\n✅ You can now ask questions about your documents!\n")

    while True:
        query = input("💬 Ask a question (or type 'exit'): ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting chat.")
            break

        if len(query) > 500:
            print("⚠️ Query is too long. Please simplify it.")
            continue

        start = time.time()

        try:
            response = qa_chain.invoke({"question": query})
            answer = response["answer"]
            print(f"\n🤖 Answer: {answer}")
            print(f"⏱️ Responded in {round(time.time() - start, 2)}s\n")

            # Auto-clear memory if too long
            if len(memory.chat_memory.messages) > 20:
                print("🧹 Clearing chat history to keep memory usage low.")
                memory.clear()

        except Exception as e:
            print(f"💥 Error: {e}\n")

if __name__ == "__main__":
    print("📄 Welcome to the GenAI Document Assistant")
    print("Enter PDF paths (comma-separated):")
    user_input = input("📁 File paths: ").strip()

    raw_paths = [path.strip() for path in user_input.split(",")]
    pdf_paths = [p for p in raw_paths if os.path.exists(p) and p.lower().endswith(".pdf")]

    if not pdf_paths:
        print("❌ No valid PDF files provided.")
        exit()

    docs = extract_text_from_multiple_pdfs(pdf_paths)

    if not any(doc["text"].strip() for doc in docs):
        print("🚫 All PDFs are empty or failed to extract text.")
        exit()

    print("🧠 Embedding documents...")
    create_vector_store_multi_doc(docs, embedding_model)
    print("✅ Vector store created.\n")

    run_chat()

    print("🧹 Cleaning up vector store...")
    shutil.rmtree("vector_store", ignore_errors=True)
    print("✅ Done.")
