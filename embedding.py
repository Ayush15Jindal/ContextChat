from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS



def chunk_text_with_metadata(text, source):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50)       
    chunks = text_splitter.split_text(text)
    metadatas = [{"source": source} for _ in chunks]
    return chunks, metadatas


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"}  
)  

def create_vector_store_multi_doc(document, embedding_model):
    all_chunks = []
    all_metadata = []
    for doc in document:
        chunks, metadata = chunk_text_with_metadata(doc["text"], doc["filename"])
        #chunks, metadata = chunk_text_with_metadata(doc["text"], doc["filename"], doc.get("page"))
        all_chunks.extend(chunks)
        all_metadata.extend(metadata)
    vector_store = FAISS.from_texts(
        all_chunks,
        embedding_model,
        metadatas=all_metadata
    )
    vector_store.save_local("vector_store")  
    

if __name__ == "__main__":
    from extract import extract_text_from_multiple_pdfs
    from extract import PDFLoader
    pdf_paths = ["COSMOS.pdf", "s12559-022-10038-y.pdf"]  
    documents = extract_text_from_multiple_pdfs(pdf_paths)
    create_vector_store_multi_doc(documents, embedding_model)
    vectorstore = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization = True)   
    
    


    
    