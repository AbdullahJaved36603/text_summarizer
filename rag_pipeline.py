from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

# Initialize tokenizer and model (Mistral-7B-Instruct)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize FAISS
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

def create_vector_store(doc_texts):
    global vector_store
    print("[INFO] Splitting and embedding documents...")

    docs = []
    for i, text in enumerate(doc_texts):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": f"doc_{i}"}))

    vector_store = FAISS.from_documents(docs, embedding)
    print(f"[INFO] Embedded {len(docs)} chunks into FAISS vector store.\n")

def summarize_with_rag(query):
    print(f"[INFO] Running RAG pipeline for query: '{query}'")
    start_time = time.time()

    # Retrieve top chunks
    retrieved_docs = vector_store.similarity_search(query, k=5)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]
    print("\n[Retrieved Chunks]")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...\n")

    # Combine chunks into a single context
    context = "\n\n".join(retrieved_chunks)

    # Token count estimation
    token_count = len(tokenizer.encode(context))
    print(f"[INFO] Estimated tokens in input: {token_count}")

    # Generate summary
    result = summarizer(f"Summarize the following:\n{context}", max_new_tokens=300, do_sample=False)[0]['generated_text']

    latency = time.time() - start_time
    print(f"[INFO] Summary generation time: {latency:.2f} seconds\n")

    return result
