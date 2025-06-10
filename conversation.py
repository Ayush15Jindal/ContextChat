from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory    
from langchain.chains import ConversationalRetrievalChain 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory
import shutil
import warnings
warnings.filterwarnings("ignore")

template = """You are an AI assistant helping users understand documents.
Use **only** the information in the provided context to answer questions.

If the answer is not found in the context, reply:
"I’m not sure based on the documents."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""

# Create a PromptTemplate object
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"}  
)  

vectorstore = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)




model = ChatOllama(model="tinyllama", temperature=0.0)



llm_for_summary = ChatOllama(model="tinyllama", temperature=0.0)

memory = ConversationSummaryBufferMemory(
    llm=llm_for_summary,
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=1000
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",  # Keep "stuff" unless too long
    retriever=retriever,
    memory=memory
)

if __name__ == "__main__":
    while True:
        query = input("Enter your question (or 'exit', 'quit', 'q' to quit): ")
        if query.lower() == "exit" or query.lower() == "quit" or query.lower() == "q":
            break
        response = qa_chain.run({"query": query})
        print(f"Response: {response}")
        
    shutil.rmtree("vector_store", ignore_errors=True)  # Clean up the vector store directory