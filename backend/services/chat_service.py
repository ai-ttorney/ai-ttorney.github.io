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
from models.crud import create_chat_message, create_chat_session, get_session_messages, get_user_sessions
from openai import OpenAI
from services.document_service import (
    conversation_chains,
    detect_section,
    document_agents,
    vector_stores,
)

# Global state
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


try:
    emsalkarar_path = os.path.join(
        os.path.dirname(__file__), "../../vectorstore/emsalkarar_faiss"
    )
    emsalkarar_store = FAISS.load_local(
        emsalkarar_path, embeddings, allow_dangerous_deserialization=True
    )
    vector_stores["emsalkarar"] = emsalkarar_store
    print("✅ Loaded emsalkarar_faiss vectorstore successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load emsalkarar_faiss vectorstore: {str(e)}")
    raise Exception(
        "Failed to load emsalkarar_faiss vectorstore. Check path or deserialization settings."
    )

llm = ChatOpenAI(
    temperature=0.5,
    model_name="ft:gpt-4o-mini-2024-07-18:aittorney::BMdnAFFk",
    max_tokens=2000,
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


def get_document_summary(doc_id: str) -> str:
    try:
        if doc_id not in vector_stores:
            return "Document not found"
        vector_store = vector_stores[doc_id]
        docs = vector_store.similarity_search("summarize", k=5)
        content = "\n".join(doc.page_content for doc in docs)

        summary_prompt = PromptTemplate(
            template="Summarize the following legal content:\n\n{content}\n\nSummary:",
            input_variables=["content"],
        )
        summary_chain = summary_prompt | llm
        return summary_chain.invoke({"content": content})
    except Exception as e:
        logger.error(f"get_document_summary error: {str(e)}")
        return f"Error: {str(e)}"


def process_chat_service(thread_id: str):
    try:
        while True:
            prompt = user_queues[thread_id].get()
            history = user_threads[thread_id]

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are AI-ttorney, a helpful legal assistant specialized in TURKISH financial law.\n"
                        "Your name is AI-ttorney, do not answer questions outside the scope of TURKISH financial law.\n"
                        "When you answer, follow these rules:\n"
                        "- Start with a clear section title.\n"
                        "- Add a relevant emoji at the beginning of each section title, based on the topic. Do NOT use the same emoji for every section. Choose an emoji that matches the topic of each section (e.g., 💼 for business, ⚖️ for law, 📅 for dates, etc.).\n"
                        "- For each item, start on a new line with a dash and a space (e.g., '- ...').\n"
                        "- Put a blank line (double line break) between the section title and the items, and between different sections.\n"
                        "- Use only plain text. Do NOT use Markdown, HTML, or any formatting symbols.\n"
                        "- Example format:\n\n"
                        "[Relevant Emoji] Section Title\n\n"
                        "- First item of the answer.\n"
                        "- Second item of the answer.\n\n"
                        "[Another Relevant Emoji] Another Section Title\n\n"
                        "- First item of the next section.\n"
                        "- Second item of the next section.\n\n"
                        "Always use clear line breaks and spacing as shown above."
                    ),
                }
            ]
            for msg in history.messages:
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})
            messages.append({"role": "user", "content": prompt})

            client = OpenAI()
            response = client.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:aittorney::BMdnAFFk",
                messages=messages,
                temperature=0.5,
                max_tokens=2000,
            )
            ai_response = response.choices[0].message.content

            print("AI RESPONSE:", repr(ai_response))

            history.add_user_message(prompt)
            history.add_ai_message(ai_response)

            db = next(get_db())
            try:
                session_id = thread_metadata[thread_id].db_session_id
                if session_id:
                    create_chat_message(db, session_id, "user", prompt)
                    create_chat_message(db, session_id, "assistant", ai_response)
                    db.commit()
            except Exception as e:
                logger.error(f"DB error in process_chat: {str(e)}")
                db.rollback()
            finally:
                db.close()

            user_queues[thread_id].task_done()
            return ai_response
    except Exception as e:
        logger.error(f"process_chat_service error: {str(e)}")
        return (
            jsonify({"error": "Unable to generate a response. Please try again."}),
            500,
        )


def generate_chat_service():
    try:
        data = request.json
        prompt = data.get("prompt")
        thread_id = data.get("thread_id")
        user_id = data.get("user_id")
        document_id = data.get("document_id")
        is_new_thread = data.get("is_new_thread", False)

        if not prompt or not user_id:
            return jsonify({"error": "Prompt and User ID required"}), 400

        if (is_new_thread or not thread_id) and prompt:
            db = next(get_db())
            try:
                title = " ".join(prompt.split()[:5]) + "..."
                new_session = create_chat_session(db, user_id, title)
                db.commit()
                thread_id = str(new_session.id)
                user_threads[thread_id] = ChatMessageHistory()
                thread_metadata[thread_id] = ThreadMetadata(thread_id, user_id)
                thread_metadata[thread_id].db_session_id = new_session.id
                thread_metadata[thread_id].title = title
            except Exception as e:
                logger.error(f"DB error on new session: {str(e)}")
                db.rollback()
                return jsonify({"error": str(e)}), 500
            finally:
                db.close()

        if thread_id not in user_threads:
            user_threads[thread_id] = ChatMessageHistory()
            thread_metadata[thread_id] = ThreadMetadata(thread_id, user_id)

        # ===== RAG AGENT PATH =====
        if document_id and (
            document_id in document_agents or document_id == "emsalkarar"
        ):
            if document_id in document_agents:
                agent = document_agents[document_id]
                response = agent.invoke({"input": prompt, "chat_history": []})
                output = response["output"]
            else:
                docs = vector_stores[document_id].similarity_search(prompt, k=5)
                combined_content = "\n\n".join(doc.page_content for doc in docs)
                summary_prompt = PromptTemplate(
                    template="Answer this query based on the following legal texts:\n\n{content}\n\nAnswer:",
                    input_variables=["content"],
                )
                rag_chain = summary_prompt | llm
                output = rag_chain.invoke({"content": combined_content})

            relevant_docs = vector_stores[document_id].similarity_search(prompt, k=5)
            sources = [
                {"content": doc.page_content, "page": doc.metadata.get("page", 0) + 1}
                for doc in relevant_docs
            ]

            db = next(get_db())
            try:
                session_id = thread_metadata[thread_id].db_session_id
                if session_id:
                    create_chat_message(db, session_id, "user", prompt)
                    create_chat_message(db, session_id, "assistant", output)
                    db.commit()
            except Exception as e:
                logger.error(f"DB error in RAG agent chat: {str(e)}")
                db.rollback()
            finally:
                db.close()

            return jsonify(
                {
                    "response": output,
                    "sources": sources,
                    "thread_id": thread_id,
                    "user_message": prompt,
                    "ai_message": output,
                }
            )

        # ===== DEFAULT CHAT PATH =====
        messages = [
            {
                "role": "system",
                "content": (
                    "You are AI-ttorney, a helpful legal assistant specialized in TURKISH financial law.\n"
                    "Your name is AI-ttorney, do not answer questions outside the scope of TURKISH financial law.\n"
                    "When you answer, follow these rules:\n"
                    "- Start with a clear section title.\n"
                    "- Add a relevant emoji at the beginning of each section title, based on the topic. Do NOT use the same emoji for every section. Choose an emoji that matches the topic of each section (e.g., 💼 for business, ⚖️ for law, 📅 for dates, etc.).\n"
                    "- For each item, start on a new line with a dash and a space (e.g., '- ...').\n"
                    "- Put a blank line (double line break) between the section title and the items, and between different sections.\n"
                    "- Use only plain text. Do NOT use Markdown, HTML, or any formatting symbols.\n"
                    "- Example format:\n\n"
                    "[Relevant Emoji] Section Title\n\n"
                    "- First item of the answer.\n"
                    "- Second item of the answer.\n\n"
                    "[Another Relevant Emoji] Another Section Title\n\n"
                    "- First item of the next section.\n"
                    "- Second item of the next section.\n\n"
                    "Always use clear line breaks and spacing as shown above."
                ),
            }
        ]
        for msg in user_threads[thread_id].messages:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})

        client = OpenAI()
        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:aittorney::BMdnAFFk",
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
        )
        ai_response = response.choices[0].message.content

        print("AI RESPONSE:", repr(ai_response))

        user_threads[thread_id].add_user_message(prompt)
        user_threads[thread_id].add_ai_message(ai_response)

        db = next(get_db())
        try:
            session_id = thread_metadata[thread_id].db_session_id
            if session_id:
                create_chat_message(db, session_id, "user", prompt)
                create_chat_message(db, session_id, "assistant", ai_response)
                db.commit()
        except Exception as e:
            logger.error(f"DB error in generate_chat_service: {str(e)}")
            db.rollback()
        finally:
            db.close()

        thread_metadata[thread_id].last_updated = datetime.now()
        thread_metadata[thread_id].last_message = ai_response

        return jsonify(
            {
                "response": ai_response,
                "thread_id": thread_id,
                "title": thread_metadata[thread_id].title,
                "user_message": prompt,
                "ai_message": ai_response,
            }
        )
    except Exception as e:
        logger.error(f"generate_chat_service error: {str(e)}")
        return (
            jsonify({"error": "Unable to generate a response. Please try again."}),
            500,
        )


def clear_chat_service():
    try:
        session.pop("messages", None)
        document_id = request.json.get("document_id")
        if document_id:
            if document_id in conversation_chains:
                conversation_chains[document_id].memory.clear()
            if document_id in document_agents:
                document_agents[document_id].memory.clear()
        return jsonify({"message": "Chat history cleared"})
    except Exception as e:
        logger.error(f"clear_chat_service error: {str(e)}")
        return (
            jsonify({"error": "Unable to clear chat history. Please try again."}),
            500,
        )
