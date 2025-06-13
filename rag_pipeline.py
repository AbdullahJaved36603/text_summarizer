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
    print(f"[INFO] Embedded {len(docs)} chunks into FAISS vector store.\n")

def summarize_with_rag(query):
    print(f"[INFO] Running RAG pipeline for query: '{query}'")
    start_time = time.time()

    # Retrieve relevant chunks from vector store
    retrieved_docs = vector_store.similarity_search(query, k=5)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]
    print("\n[Retrieved Chunks]")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...\n")

    # Build context from top-k chunks
    context = "\n\n".join(retrieved_chunks)
    token_count = len(tokenizer.encode(context))
    print(f"[INFO] Estimated tokens in input: {token_count}")

    # Prepare prompt
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="{context}"  # Note: BART expects raw text for summarization
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate summary
    summary = chain.run(context)

    latency = time.time() - start_time
    print(f"[INFO] Summary generation time: {latency:.2f} seconds\n")
    return summary
