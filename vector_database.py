import os
import fitz  # PyMuPDF
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def safe_embedding_call(texts, embed_model):
    return embed_model.embed_documents(texts)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pdf_dir = "E:\AITTORNEY\RAG"
vectorstore_dir = "vectorstore/emsalkarar_faiss"

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

pdf_files = [os.path.join(pdf_dir, file) for file in os.listdir(pdf_dir) if file.endswith(".pdf")]

all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for file in pdf_files:
    raw_text = extract_text_from_pdf(file)
    chunks = text_splitter.split_text(raw_text)
    docs = [Document(page_content=chunk, metadata={"source": os.path.basename(file)}) for chunk in chunks]
    all_chunks.extend(docs)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_db = FAISS.from_documents(all_chunks, embeddings)
vector_db.save_local(vectorstore_dir)

print(f"✅ {vectorstore_dir} içine {len(all_chunks)} doküman kaydedildi.")
