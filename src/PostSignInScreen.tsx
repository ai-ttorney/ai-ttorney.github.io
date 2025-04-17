import { useState, useEffect } from 'react';
import { UserButton, useUser } from "@clerk/clerk-react";

interface Thread {
  thread_id: string;
  title: string;
  created_at: string;
  last_updated: string;
  last_message?: string;
  is_initial: boolean;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

function PostSignInScreen() {
  const { user } = useUser();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThread, setActiveThread] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isInitialChat, setIsInitialChat] = useState(true);
  const [hasInitialInput, setHasInitialInput] = useState(false);
  const [isBackendDown, setIsBackendDown] = useState(false);

  const API_BASE_URL = 'http://127.0.0.1:5000';

  // Add this useEffect to load initial messages
  useEffect(() => {
    const loadInitialMessages = async () => {
      if (!user) return;
      
      try {
        // Try to fetch threads from backend
        const response = await fetch(`${API_BASE_URL}/get_all_threads?user_id=${user.id}`);
        const data = await response.json();
        setThreads(data.threads || []);
        setIsBackendDown(false);
      } catch (error) {
        console.error("Backend is down, loading from local storage");
        setIsBackendDown(true);
        
        // Load all threads from local storage
        const savedThreads = localStorage.getItem('chat_threads');
        if (savedThreads) {
          setThreads(JSON.parse(savedThreads));
        }
      }
    };

    loadInitialMessages();
  }, [user]);

  const fetchThreads = async () => {
    if (!user) return;
    try {
      const response = await fetch(`${API_BASE_URL}/get_all_threads?user_id=${user.id}`);
      const data = await response.json();
      setThreads(data.threads || []);
      // Save threads to local storage
      localStorage.setItem('chat_threads', JSON.stringify(data.threads || []));
    } catch (error) {
      console.error("Error fetching threads:", error);
      // If backend is down, try to load from local storage
      const savedThreads = localStorage.getItem('chat_threads');
      if (savedThreads) {
        setThreads(JSON.parse(savedThreads));
      }
    }
  };

  useEffect(() => {
    if (user) {
      fetchThreads();
    }
  }, [user]);

  const createNewThread = async () => {
    if (!user) return;
    try {
      const response = await fetch(`${API_BASE_URL}/create_thread`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: user.id })
      });
      const data = await response.json();
      setActiveThread(data.thread_id);
      setMessages([]);
      setInputMessage('');
      setIsInitialChat(false);
      setHasInitialInput(false);
      await fetchThreads();
    } catch (error) {
      console.error("Error creating thread:", error);
      // If backend is down, create a local thread
      const localThreadId = `local_${Date.now()}`;
      setActiveThread(localThreadId);
      setMessages([]);
      setInputMessage('');
      setIsInitialChat(false);
      setHasInitialInput(false);
      
      // Add to local threads
      const newThread = {
        thread_id: localThreadId,
        title: "New Chat",
        created_at: new Date().toISOString(),
        last_updated: new Date().toISOString(),
        is_initial: true
      };
      setThreads(prev => [...prev, newThread]);
      localStorage.setItem('chat_threads', JSON.stringify([...threads, newThread]));
    }
  };

  const deleteThread = async (threadId: string) => {
    if (!user) return;
    try {
      console.log('Attempting to delete thread:', threadId);
      const response = await fetch(`${API_BASE_URL}/delete_thread`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          thread_id: threadId,
          user_id: user.id
        }),
      });

      const data = await response.json();
      
      if (response.ok) {
        console.log('Thread deleted successfully:', threadId);
        // Remove the thread from the local state
        setThreads(prevThreads => prevThreads.filter(thread => thread.thread_id !== threadId));
        
        // If the deleted thread was active, clear the active thread
        if (activeThread === threadId) {
          setActiveThread(null);
          setMessages([]);
        }
      } else {
        console.error('Failed to delete thread:', data.error);
        alert(`Failed to delete thread: ${data.error}`);
      }
    } catch (error) {
      console.error('Error deleting thread:', error);
      alert('An error occurred while deleting the thread. Please try again.');
    }
  };

  const loadThread = async (threadId: string) => {
    if (!user) return;
    setIsLoading(true);
    try {
      setActiveThread(threadId);
      const response = await fetch(`${API_BASE_URL}/get_thread_history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: threadId,
          user_id: user.id
        }),
      });
      const data = await response.json();
      const loadedMessages = data.history || [];
      setMessages(loadedMessages);
      // Save to local storage
      localStorage.setItem(`chat_messages_${threadId}`, JSON.stringify(loadedMessages));
      setIsInitialChat(false);
      setInputMessage('');
    } catch (error) {
      console.error("Error loading thread:", error);
      // If backend is down, try to load from local storage
      const savedMessages = localStorage.getItem(`chat_messages_${threadId}`);
      if (savedMessages) {
        setMessages(JSON.parse(savedMessages));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !user) return;

    const message = inputMessage.trim();
    setInputMessage('');

    let currentThreadId = activeThread;

    // If this is the initial chat, create a new thread
    if (isInitialChat && !currentThreadId) {
      try {
        const response = await fetch(`${API_BASE_URL}/create_thread`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: user.id })
        });
        const data = await response.json();
        currentThreadId = data.thread_id;
        setActiveThread(currentThreadId);
        setIsInitialChat(false);
        setHasInitialInput(true);
      } catch (error) {
        console.error("Error creating initial thread:", error);
        return;
      }
    }

    // Add user message to UI immediately
    const newUserMessage: Message = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    };
    const updatedMessages = [...messages, newUserMessage];
    setMessages(updatedMessages);
    // Save to local storage
    localStorage.setItem(`chat_messages_${currentThreadId}`, JSON.stringify(updatedMessages));

    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: message,
          thread_id: currentThreadId,
          user_id: user.id,
          is_new_thread: isInitialChat
        }),
      });
      const data = await response.json();

      // Only add the AI response if we're still on the same thread
      if (currentThreadId === activeThread) {
        const newAssistantMessage: Message = {
          role: 'assistant',
          content: data.ai_message,
          timestamp: new Date().toISOString()
        };
        const finalMessages = [...updatedMessages, newAssistantMessage];
        setMessages(finalMessages);
        // Save to local storage
        localStorage.setItem(`chat_messages_${currentThreadId}`, JSON.stringify(finalMessages));
      }

      // Update the thread title in the sidebar if it's a new thread
      if (isInitialChat && data.title) {
        setThreads(prevThreads => 
          prevThreads.map(thread => 
            thread.thread_id === currentThreadId 
              ? { ...thread, title: data.title }
              : thread
          )
        );
      }

      // Refresh threads to show updated title and messages
      await fetchThreads();
    } catch (error) {
      console.error("Error sending message:", error);
      // If backend is down, we already have the messages in local storage
    }
  };

  useEffect(() => {
    // Dynamically load the CSS and JS files
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/styles.css';
    document.head.appendChild(link);

    const script = document.createElement('script');
    script.src = '/script.js';
    script.defer = true;
    document.body.appendChild(script);

    return () => {
      // Cleanup the injected resources when the component unmounts
      document.head.removeChild(link);
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <button onClick={createNewThread} className="new-thread-btn">+ New Chat</button>

        <div className="thread-list">
          {threads.map((thread) => (
            <div
              key={thread.thread_id}
              className={`thread-item ${thread.thread_id === activeThread ? 'active' : ''}`}
            >
              <div className="thread-content" onClick={() => loadThread(thread.thread_id)}>
                <div className="thread-title">{thread.title}</div>
                <div className="thread-metadata">
                  {new Date(thread.last_updated).toLocaleDateString()}
                </div>
              </div>
              <button
                className="delete-thread-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  if (window.confirm('Are you sure you want to delete this chat?')) {
                    deleteThread(thread.thread_id);
                  }
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>

        <div className="user-profile">
          {user ? (
            <div className="profile-content">
              <p>Welcome, {user.firstName || user.username}!</p>
              <UserButton />
            </div>
          ) : (
            <p>Please sign in</p>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="main-content">
        <button
          className="sidebar-toggle"
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        >
          {isSidebarOpen ? '✖' : '☰'}
        </button>

        <div className="chat-container">
          {isLoading ? (
            <div className="loading">Loading...</div>
          ) : (
            <>
              {isInitialChat && !hasInitialInput && (
                <div className="initial-chat-message">
                  <h2>Welcome to AI-ttorney!</h2>
                  <p>Start a new conversation by typing your message below.</p>
                </div>
              )}
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
                >
                  <div className="message-content">{message.content}</div>
                  <div className="message-timestamp">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

        <div className="message-input-container">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder={isInitialChat ? "Type your first message here..." : "Type your message here..."}
            className={isInitialChat ? 'initial-chat-input' : ''}
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim()}
            className={isInitialChat ? 'initial-chat-button' : ''}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default PostSignInScreen;
