import uuid
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
import tempfile
import traceback
import logging
import threading
import queue
from datetime import datetime
from bson import ObjectId
import sys
from urllib.parse import quote_plus
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sqlalchemy.orm import Session
from database import get_db
from models import ChatSession, ChatMessage
from crud import create_chat_session, create_chat_message, get_user_sessions, get_session_messages


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OpenAI API key not found in environment variables")
    sys.exit(1)

# Initialize LangChain components
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(
        temperature=0.5,
        model_name="ft:gpt-4o-mini-2024-07-18:aittorney::BMdnAFFk",
    )
except Exception as e:
    logger.error(f"Error initializing LangChain components: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Store conversation chains, vector stores, and agents
conversation_chains = {}
vector_stores = {}
document_agents = {}

# Thread management
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

# In-memory storage for active sessions
user_threads = {}
user_queues = {}
thread_metadata = {}

# RAG agent 
@tool
def search_document(query: str, doc_id: str) -> str:
    """Search for information in the document using the query."""
    try:
        if doc_id not in vector_stores:
            return "Document not found"
        
        vector_store = vector_stores[doc_id]
        docs = vector_store.similarity_search(query, k=3)
        
        results = []
        for doc in docs:
            results.append(f"Page {doc.metadata.get('page', 0) + 1}: {doc.page_content}")
            
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"Error in search_document: {str(e)}")
        return f"Error searching document: {str(e)}"

@tool
def get_document_summary(doc_id: str) -> str:
    """Get a summary of the entire document."""
    try:
        if doc_id not in vector_stores:
            return "Document not found"
        
        vector_store = vector_stores[doc_id]
        docs = vector_store.similarity_search("Give me a summary of this document", k=5)
        
        content = "\n".join([doc.page_content for doc in docs])
        
        summary_prompt = PromptTemplate(
            template="Please provide a concise summary of the following document content:\n\n{content}\n\nSummary:",
            input_variables=["content"]
        )
        
        summary_chain = summary_prompt | llm
        return summary_chain.invoke({"content": content})
    except Exception as e:
        logger.error(f"Error in get_document_summary: {str(e)}")
        return f"Error getting document summary: {str(e)}"


def create_document_agent(doc_id: str):
    tools = [
        Tool(
            name="SearchDocument",
            func=lambda q: search_document(q, doc_id),
            description="Search for specific information in the document"
        ),
        Tool(
            name="GetDocumentSummary",
            func=lambda: get_document_summary(doc_id),
            description="Get a summary of the entire document"
        )
    ]
    
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    )
    
# RAG vector database
try:
    predefined_path = "vectorstore/emsalkarar_faiss"
    if os.path.exists(predefined_path):
        vector_store = FAISS.load_local(predefined_path, embeddings, allow_dangerous_deserialization=True)
        vector_stores["emsalkarar"] = vector_store
        document_agents["emsalkarar"] = create_document_agent("emsalkarar")
        print("✅ EmsalKarar vectorstore yüklendi.")
except Exception as e:
    print(f"🚨 EmsalKarar vectorstore yüklenemedi: {e}")

@app.route('/process_chat', methods=['POST'])
def process_chat(thread_id: str):
    """Process chat messages and generate AI responses"""
    while True:
        prompt = user_queues[thread_id].get()
        history = user_threads[thread_id]
        history.add_user_message(prompt)
        
        messages = [{"role": "system", "content": "You are AI-ttorney, an AI assistant trained only on Turkish financial law. You should politely decline to answer questions outside this field, such as criminal, family, or civil law, by explaining that you're only able to assist with Turkish financial legal matters. You can vary your wording. Here are a few examples of how you might respond: - 'Bu konuda yardımcı olamıyorum çünkü yalnızca Türkiye finansal hukuku üzerine eğitildim.' - 'Üzgünüm, yalnızca Türkiye'nin finansal hukuku hakkında bilgi verebiliyorum.' - 'Bu konu benim uzmanlık alanıma girmiyor; finansal hukuk hakkında sorularınızı yanıtlayabilirim.' If the user greets you (e.g., 'selam', 'merhaba'), you can greet them back and remind them of your scope."}]
        
        for msg in history.messages:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})
        
        try:
            response = openai.ChatCompletion.create(
                model="ft:gpt-4o-mini-2024-07-18:aittorney::BMdnAFFk",
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
            ai_response = response['choices'][0]['message']['content']
            history.add_ai_message(ai_response)
            
        # Store messages in the database
            db = next(get_db())
            try:
                # Get the session ID from thread metadata
                session_id = thread_metadata[thread_id].db_session_id
                if not session_id:
                    logger.error(f"No session ID found for thread {thread_id}")
                    return ai_response
                
                # Store user message
                create_chat_message(db, session_id, "user", prompt)
                # Store AI response
                create_chat_message(db, session_id, "assistant", ai_response)
                
                db.commit()
            except Exception as e:
                logger.error(f"Error storing messages in database: {str(e)}")
                db.rollback()
            finally:
                db.close()
        except Exception as e:
            ai_response = f"Error: {str(e)}"
            logger.error(f"Error in process_chat: {str(e)}")
        
        user_queues[thread_id].task_done()
        return ai_response



# file upload için
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_document():
    try:
        logger.info("Starting file upload process")
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.pdf'):
            logger.warning("Invalid file type")
            return jsonify({"error": "Only PDF files are supported"}), 400

        logger.info(f"Processing file: {file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            try:
                file.save(temp_file.name)
                logger.info(f"File saved to temporary location: {temp_file.name}")
                
                # Load and process the PDF
                try:
                    loader = PyPDFLoader(temp_file.name)
                    pages = loader.load()
                    logger.info(f"PDF loaded successfully: {len(pages)} pages")
                except Exception as e:
                    logger.error(f"Error loading PDF: {str(e)}")
                    return jsonify({"error": f"Error loading PDF: {str(e)}"}), 500

                # Split the document
                try:
                    text_splitter = CharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100,
                        separator="\n"
                    )
                    chunks = text_splitter.split_documents(pages)
                    logger.info(f"Document split into {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error splitting document: {str(e)}")
                    return jsonify({"error": f"Error splitting document: {str(e)}"}), 500

                # Create vector store
                try:
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    vector_stores[file.filename] = vector_store
                    logger.info("Vector store created successfully")
                except Exception as e:
                    logger.error(f"Error creating vector store: {str(e)}")
                    return jsonify({"error": f"Error creating vector store: {str(e)}"}), 500

                # Create conversation chain and agent
                try:
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                    
                    conversation_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                        memory=memory,
                        return_source_documents=True
                    )
                    conversation_chains[file.filename] = conversation_chain
                    
                    # Create and store document agent
                    document_agents[file.filename] = create_document_agent(file.filename)
                    logger.info("Conversation chain and agent created successfully")
                except Exception as e:
                    logger.error(f"Error creating conversation chain or agent: {str(e)}")
                    return jsonify({"error": f"Error creating conversation chain or agent: {str(e)}"}), 500

            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": f"Error processing file: {str(e)}"}), 500
            finally:
                try:
                    os.unlink(temp_file.name)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {str(e)}")

            return jsonify({
                "message": "Document processed successfully",
                "document_id": file.filename
            })

    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    try:
    data = request.json
    prompt = data.get("prompt", "")
        thread_id = data.get("thread_id")
        user_id = data.get("user_id")
        document_id = data.get("document_id")
        is_new_thread = data.get("is_new_thread", False)
        
        logger.info(f"Generate request - prompt: {prompt}, thread_id: {thread_id}, user_id: {user_id}, document_id: {document_id}")
        
    if not prompt:
            logger.warning("No prompt provided")
        return jsonify({"error": "Prompt is required"}), 400

        if not user_id:
            logger.warning("No user ID provided")
            return jsonify({"error": "User ID is required"}), 400

        # Create new thread if requested or if no thread_id exists
        if is_new_thread or not thread_id:
            # Create new database session
            db = next(get_db())
            try:
                # Use first few words of the prompt as the title
                title = " ".join(prompt.split()[:5]) + "..."
                new_session = create_chat_session(db, user_id, title)
                db.commit()
                
                # Use the database session ID as the thread ID
                thread_id = str(new_session.id)
                
                # Initialize thread metadata
                user_threads[thread_id] = ChatMessageHistory()
                thread_metadata[thread_id] = ThreadMetadata(thread_id, user_id)
                thread_metadata[thread_id].db_session_id = new_session.id
                thread_metadata[thread_id].title = title
                
            except Exception as e:
                logger.error(f"Error creating database session: {str(e)}")
                db.rollback()
                return jsonify({"error": str(e)}), 500
            finally:
                db.close()
        else:
            # Get existing session from database
            db = next(get_db())
            try:
                session = db.query(ChatSession).filter(
                    ChatSession.id == int(thread_id),
                    ChatSession.user_id == user_id
                ).first()
                if not session:
                    return jsonify({"error": "Thread not found"}), 404
                
                # Load existing messages from database
                messages = get_session_messages(db, session.id)
                if thread_id not in user_threads:
                    user_threads[thread_id] = ChatMessageHistory()
                if thread_id not in thread_metadata:
                    thread_metadata[thread_id] = ThreadMetadata(thread_id, user_id)
                
                # Add messages to chat history
                for msg in messages:
                    if msg.role == "user":
                        user_threads[thread_id].add_user_message(msg.content)
                    else:
                        user_threads[thread_id].add_ai_message(msg.content)
                
                thread_metadata[thread_id].db_session_id = session.id
                thread_metadata[thread_id].title = session.session_name
            except Exception as e:
                logger.error(f"Error getting database session: {str(e)}")
                return jsonify({"error": str(e)}), 500
            finally:
                db.close()
        
        # Handle document-specific queries
        if document_id and document_id in document_agents:
            logger.info("Using document-specific agent")
            try:
                agent = document_agents[document_id]
                response = agent.invoke({
                    "input": prompt,
                    "chat_history": [],
                    "system_message": "You are AI-ttorney, an AI assistant trained only on Turkish financial law. You should politely decline to answer questions outside this field, such as criminal, family, or civil law, by explaining that you're only able to assist with Turkish financial legal matters. You can vary your wording. Here are a few examples of how you might respond: - 'Bu konuda yardımcı olamıyorum çünkü yalnızca Türkiye finansal hukuku üzerine eğitildim.' - 'Üzgünüm, yalnızca Türkiye'nin finansal hukuku hakkında bilgi verebiliyorum.' - 'Bu konu benim uzmanlık alanıma girmiyor; finansal hukuk hakkında sorularınızı yanıtlayabilirim.' If the user greets you (e.g., 'selam', 'merhaba'), you can greet them back and remind them of your scope."

                })
                
                vector_store = vector_stores[document_id]
                relevant_docs = vector_store.similarity_search(prompt, k=3)
                
                sources = []
                for doc in relevant_docs:
                    sources.append({
                        "content": doc.page_content,
                        "page": doc.metadata.get("page", 0) + 1
                    })
                
                return jsonify({
                    "response": response["output"],
                    "sources": sources,
                    "thread_id": thread_id,
                    "user_message": prompt,
                    "ai_message": response["output"]
                })
            except Exception as e:
                logger.error(f"Error in document processing with agent: {str(e)}")
                return jsonify({"error": f"Error processing document: {str(e)}"}), 500

        # Ensure thread exists
        if thread_id not in user_threads:
            user_threads[thread_id] = ChatMessageHistory()
            
            thread_metadata[thread_id] = ThreadMetadata(thread_id, user_id)
        
        # Verify thread belongs to user
        if thread_metadata[thread_id].user_id != user_id:
            return jsonify({"error": "Unauthorized access to thread"}), 403
        
        # Add user message to chat history
        user_threads[thread_id].add_user_message(prompt)
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": "AI-ttorney is a financial law advice chatbot and will only answer questions related to Turkey's financial law. For all other questions, it must say that it is only trained for Turkey's financial law."}
        ]

        # Add chat history
        for msg in user_threads[thread_id].messages:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})
        
    try:
            # Get AI response
        response = openai.ChatCompletion.create(
                model="ft:gpt-4o-mini-2024-07-18:aittorney::BMdnAFFk",
                messages=messages,
            max_tokens=100
        )
            ai_response = response['choices'][0]['message']['content']
            
            # Add AI response to chat history
            user_threads[thread_id].add_ai_message(ai_response)
            
            # Store messages in database
            db = next(get_db())
            try:
                session_id = thread_metadata[thread_id].db_session_id
                if session_id:
                    # Store user message
                    create_chat_message(db, session_id, "user", prompt)
                    # Store AI response
                    create_chat_message(db, session_id, "assistant", ai_response)
                    db.commit()
            except Exception as e:
                logger.error(f"Error storing messages in database: {str(e)}")
                db.rollback()
            finally:
                db.close()
        
        # Update thread metadata
            thread_metadata[thread_id].last_updated = datetime.now()
            thread_metadata[thread_id].last_message = ai_response
            thread_metadata[thread_id].is_initial = False
        
        # Update title only for the first message in a new thread
        # Update title only for the first message in a new thread
            if thread_metadata[thread_id].title == "New Chat":
                title = " ".join(prompt.split()[:5]) + "..."
                thread_metadata[thread_id].title = title
                
                # Update session name in database
                db = next(get_db())
                try:
                    session = db.query(ChatSession).filter(
                        ChatSession.id == thread_metadata[thread_id].db_session_id
                    ).first()
                    if session:
                        session.session_name = title
                        db.commit()
                except Exception as e:
                    logger.error(f"Error updating session name: {str(e)}")
                    db.rollback()
                finally:
                    db.close()
        
            return jsonify({
                "response": ai_response,
                "thread_id": thread_id,
                "title": thread_metadata[thread_id].title,
                "user_message": prompt,
                "ai_message": ai_response
            })
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/create_thread', methods=['POST'])
def create_thread():
    """Create a new thread for a user"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        
        # Create new database session
        db = next(get_db())
        try:
            new_session = create_chat_session(db, user_id, "New Chat")
            db.commit()
            
            # Use the database session ID as the thread ID
            thread_id = str(new_session.id)
            
            # Initialize thread metadata
            user_threads[thread_id] = ChatMessageHistory()
            user_queues[thread_id] = queue.Queue()
            thread_metadata[thread_id] = ThreadMetadata(thread_id, user_id)
            thread_metadata[thread_id].db_session_id = new_session.id
            thread_metadata[thread_id].title = "New Chat"
        
            return jsonify({
                "thread_id": thread_id,
                "title": "New Chat",
                "created_at": new_session.created_at.isoformat(),
                "last_updated": new_session.updated_at.isoformat(),
                "is_initial": True
            })

        except Exception as e:
            logger.error(f"Error creating thread: {str(e)}")
            logger.error(traceback.format_exc())
            db.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in create_thread: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/get_all_threads', methods=['GET'])
def get_all_threads():
    """Get all threads for a specific user"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Get sessions from database
    db = next(get_db())
    try:
        sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
        threads_data = []
        
        for session in sessions:
            # Get the last message for this session
            last_message = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).order_by(ChatMessage.timestamp.desc()).first()
            
            threads_data.append({
                'thread_id': str(session.id),
                'title': session.session_name or "New Chat",
                'created_at': session.created_at.isoformat(),
                'last_updated': session.updated_at.isoformat(),
                'last_message': last_message.content if last_message else None,
                'is_initial': not last_message
            })
            
        # Sort by last updated time, newest first
        threads_data.sort(key=lambda x: x['last_updated'], reverse=True)
        return jsonify({'threads': threads_data})
        
    except Exception as e:
        logger.error(f"Error getting threads from database: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/get_thread_history', methods=['POST'])
def get_thread_history():
    """Get chat history for a specific thread"""
    data = request.json
    thread_id = data.get('thread_id')
    user_id = data.get('user_id')
    
    if not thread_id or not user_id:
        return jsonify({"error": "Thread ID and User ID are required"}), 400
    
    db = next(get_db())
    try:
        # Get session from database
        session = db.query(ChatSession).filter(
            ChatSession.id == int(thread_id),
            ChatSession.user_id == user_id
        ).first()
        
        if not session:
            return jsonify({"error": "Thread not found"}), 404
            
        # Get all messages for this session
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session.id
        ).order_by(ChatMessage.timestamp.asc()).all()
        
        # Convert messages to the required format
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            })
            
        # Update the in-memory history
        if thread_id in user_threads:
            user_threads[thread_id].clear()
            for msg in history:
                if msg["role"] == "user":
                    user_threads[thread_id].add_user_message(msg["content"])
                else:
                    user_threads[thread_id].add_ai_message(msg["content"])
        
        return jsonify({
            "history": history,
            "title": session.session_name or "New Chat",
            "thread_id": thread_id
        })

    except Exception as e:
        logger.error(f"Error getting thread history from database: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/delete_thread', methods=['POST'])
def delete_thread():
    """Delete a thread for a specific user"""
    data = request.json
    thread_id = data.get('thread_id')
    user_id = data.get('user_id')
    
    if not thread_id or not user_id:
        return jsonify({"error": "Thread ID and User ID are required"}), 400
    
    db = next(get_db())
    try:
        # Get session from database
        session = db.query(ChatSession).filter(
            ChatSession.id == int(thread_id),
            ChatSession.user_id == user_id
        ).first()
        
        if not session:
            return jsonify({"error": "Thread not found"}), 404
            
        # Delete the session (this will cascade delete messages due to the relationship)
        db.delete(session)
        db.commit()
        
        # Clean up in-memory data
        if thread_id in user_threads:
            del user_threads[thread_id]
        if thread_id in user_queues:
            del user_queues[thread_id]
        if thread_id in thread_metadata:
            del thread_metadata[thread_id]
        
        return jsonify({"message": "Thread deleted successfully"})
        
    except Exception as e:
        logger.error(f"Error deleting thread: {str(e)}")
        logger.error(traceback.format_exc())
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/clear', methods=['POST'])
def clear():
    try:
    session.pop('messages', None)
        document_id = request.json.get("document_id")
        if document_id:
            if document_id in conversation_chains:
                conversation_chains[document_id].memory.clear()
            if document_id in document_agents:
                document_agents[document_id].memory.clear()
    return jsonify({"message": "Conversation history cleared."})
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
