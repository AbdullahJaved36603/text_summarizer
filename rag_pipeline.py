from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

# Load summarization model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
pipe = pipeline("summarization", model=model, tokenizer=tokenizer, max_length=200, min_length=30, do_sample=False)

# Wrap into LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Embedding model for FAISS
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
    print(f"[INFO] Total number of chunks created: {len(docs)}")
    print(f"[INFO] Embedded {len(docs)} chunks into FAISS vector store.\n")

def summarize_with_rag(query):
    print(f"[INFO] Running RAG pipeline for query: '{query}'")
    start_time = time.time()

    # Total chunks in vector store
    total_chunks = len(vector_store.index_to_docstore_id)
    k = max(1, total_chunks // 3)  # At least retrieve 1
    print(f"[INFO] Retrieving top {k} chunks out of {total_chunks}")

    # Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(query, k=k)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]
    print("\n[Retrieved Chunks]")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...\n")

    # Build context
    context = "\n\n".join(retrieved_chunks)
    input_tokens = len(tokenizer.encode(context, truncation=True, max_length=1024))
    print(f"[INFO] Estimated input tokens: {input_tokens}")

    # Ensure input tokens are within BART's max limit (1024)
    if input_tokens > 950:
        context = tokenizer.decode(tokenizer.encode(context, truncation=True, max_length=950), skip_special_tokens=True)
        input_tokens = len(tokenizer.encode(context))
        print(f"[INFO] Context truncated to {input_tokens} tokens")

    # Prepare tokenizer input
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=950)

    # Dynamically set max output length
    max_total_tokens = 1024
    available_output_tokens = max_total_tokens 
    output_max_length =input_tokens
  #  output_max_length = max(min(available_output_tokens/3,1), min(available_output_tokens, 950))  
    print(output_max_length)
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max(25,output_max_length),
        min_length=max(25,int(output_max_length/4)),
        length_penalty=1.90,
        num_beams=4,
        early_stopping=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    latency = time.time() - start_time
    print(f"[INFO] Summary generation time: {latency:.2f} seconds\n")
    return summary
