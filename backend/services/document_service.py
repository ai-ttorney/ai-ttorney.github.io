import logging
import os
import queue
import tempfile
import traceback
from datetime import datetime

from database.database import get_db
from dotenv import load_dotenv
from flask import jsonify, request, session
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from models.crud import create_chat_message, create_chat_session, get_session_messages
from openai import OpenAI

conversation_chains = {}
vector_stores = {}
document_agents = {}
user_threads = {}
user_queues = {}
thread_metadata = {}

load_dotenv()
logger = logging.getLogger("__name__")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API key not found")
    raise Exception("Missing OpenAI key")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(
    temperature=0.5,
    model_name="ft:gpt-4o-mini-2024-07-18:aittorney::BMdnAFFk",
    max_tokens=1000,
)


class ThreadMetadata:
    def __init__(self, thread_id: str, user_id: str):
        self.thread_id = thread_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.title = "New Conversation"
        self.last_message = None
        self.is_initial = True
        self.db_session_id = None


def create_document_agent(doc_id: str):
    tools = [
        Tool(
            name="SearchDocument",
            func=lambda q: search_document(q, doc_id),
            description="Search for specific legal info in the document",
        ),
        Tool(
            name="GetDocumentSummary",
            func=lambda: get_document_summary(doc_id),
            description="Summarize the document",
        ),
    ]
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=ChatMessageHistory(),
    )


def search_document(query: str, doc_id: str) -> str:
    try:
        if doc_id not in vector_stores:
            return "Document not found"
        vector_store = vector_stores[doc_id]

        section_filter = None
        if "iflas" in query.lower():
            section_filter = "Bankruptcy Law"
        elif "haciz" in query.lower():
            section_filter = "Execution Law"
        elif "borçlu" in query.lower():
            section_filter = "Debtor Law"

        if section_filter:
            docs = vector_store.similarity_search(
                query, k=5, filter={"section": section_filter}
            )
        else:
            docs = vector_store.similarity_search(query, k=5)

        results = [
            f"Page {doc.metadata.get('page', 0) + 1}: {doc.page_content.strip()}"
            for doc in docs
        ]
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"search_document error: {str(e)}")
        return f"Error: {str(e)}"


def detect_section(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["banka", "mevduat", "bankacılık"]):
        return "Bankacılık Hukuku"
    elif any(k in lower for k in ["sermaye piyasası", "spk", "halka arz"]):
        return "Sermaye Piyasası Hukuku"
    elif any(k in lower for k in ["menkul", "hisse senedi", "tahvil"]):
        return "Menkul Kıymetler Hukuku"
    elif any(k in lower for k in ["bist", "borsa", "istanbul menkul"]):
        return "Borsa İstanbul Hukuku"
    elif any(k in lower for k in ["leasing", "finansal kiralama"]):
        return "Finansal Kiralama Hukuku"
    elif any(k in lower for k in ["kredi", "ipotek", "teminat", "rehin"]):
        return "Kredi ve Teminat Hukuku"
    elif any(k in lower for k in ["vergi usul", "vuk", "beyanname"]):
        return "Vergi Usul Hukuku"
    elif any(k in lower for k in ["gelir vergisi", "stopaj", "gelir"]):
        return "Gelir Vergisi Hukuku"
    elif any(k in lower for k in ["kurumlar vergisi", "kurum kazancı"]):
        return "Kurumlar Vergisi Hukuku"
    elif any(k in lower for k in ["amme alacağı", "6183", "ödeme emri"]):
        return "Amme Alacakları Hukuku"
    elif any(k in lower for k in ["finansal", "faiz", "döviz"]):
        return "Finansal Hukuk"
    else:
        return "Genel Hukuk"


def upload_document_service():
    try:
        logger.info("Starting file upload")

        if "file" not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith(".pdf"):
            logger.warning("Invalid file type")
            return jsonify({"error": "Only PDF files allowed"}), 400

        logger.info(f"Processing file: {file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            try:
                file.save(temp_file.name)
                logger.info(f"Saved temp file: {temp_file.name}")

                loader = PyPDFLoader(temp_file.name)
                pages = loader.load()
                logger.info(f"PDF loaded: {len(pages)} pages")

                text_splitter = CharacterTextSplitter(
                    chunk_size=512, chunk_overlap=100, separator=" "
                )
                chunks = text_splitter.split_documents(pages)
                logger.info(f"Split into {len(chunks)} chunks")

                for i, doc in enumerate(chunks):
                    doc.metadata["chunk_index"] = i
                    doc.metadata["source_file"] = file.filename
                    doc.metadata["page"] = doc.metadata.get("page", i // 2)
                    doc.metadata["section"] = detect_section(doc.page_content)

                vector_store = FAISS.from_documents(chunks, embeddings)
                vector_stores[file.filename] = vector_store
                logger.info("Vector store created")

                memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )
                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                    memory=memory,
                    return_source_documents=True,
                )
                conversation_chains[file.filename] = conversation_chain
                document_agents[file.filename] = create_document_agent(file.filename)
                logger.info("RAG components initialized")

            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
            finally:
                try:
                    os.unlink(temp_file.name)
                    logger.info("Cleaned up temporary file")
                except Exception as e:
                    logger.warning(f"Temp file cleanup failed: {str(e)}")

            return jsonify(
                {
                    "message": "Document processed successfully",
                    "document_id": file.filename,
                }
            )

    except Exception as e:
        logger.error(f"Unhandled error in upload_document_service: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
