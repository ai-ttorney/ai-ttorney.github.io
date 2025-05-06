import { useState, useEffect } from 'react';
import { fetchAllThreads, deleteThreadById, getThreadHistory, sendMessageToBackend } from '../services/apiService';
import { applyDarkMode, loadDynamicResources } from '../utils/themeManager';
import { translations } from '../utils/translations';
import { generateChatHistoryPDF } from '../utils/downloadChatHistory';
import { Thread, Message, Language } from '../types/chatTypes';
import { useUser } from '@clerk/clerk-react';

export const useChat = () => {
  const { user } = useUser();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });
  const [language, setLanguage] = useState<Language>(() => {
    const saved = localStorage.getItem('language');
    return (saved as Language) || 'en';
  });
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThread, setActiveThread] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isInitialChat, setIsInitialChat] = useState(true);
  const [hasInitialInput, setHasInitialInput] = useState(false);
  const [isBackendDown, setIsBackendDown] = useState(false);

  const t = translations[language];

  useEffect(() => {
    if (isDarkMode) {
      applyDarkMode(true);
    }
    const cleanup = loadDynamicResources();
    return cleanup;
  }, []);

  useEffect(() => {
    const loadThreads = async () => {
      if (!user) return;
      try {
        const data = await fetchAllThreads(user.id);
        setThreads(data.threads || []);
        setIsBackendDown(false);
      } catch (error) {
        console.error('Backend is down');
        setIsBackendDown(true);
      }
    };
    loadThreads();
  }, [user]);

  const refreshThreads = async () => {
    if (!user) return;
    try {
      const data = await fetchAllThreads(user.id);
      setThreads(data.threads || []);
    } catch (error) {
      console.error('Error refreshing threads');
      setIsBackendDown(true);
    }
  };

 
  const handleCreateThread = () => {
    if (!user || isBackendDown) return;
    setActiveThread(null);
    setMessages([]);
    setInputMessage('');
    setIsInitialChat(true);
    setHasInitialInput(false);
  };

  const handleDeleteThread = async (threadId: string) => {
    if (!user) return;
    try {
      const response = await deleteThreadById(threadId, user.id);
      if (response.ok) {
        setThreads(prev => prev.filter(thread => thread.thread_id !== threadId));
        if (activeThread === threadId) {
          setActiveThread(null);
          setMessages([]);
        }
      } else {
        console.error('Failed to delete thread');
      }
    } catch (error) {
      console.error('Error deleting thread');
    }
  };

  const handleLoadThread = async (threadId: string) => {
    if (!user) return;
    setIsLoading(true);
    setErrorMessage('');
    try {
      setActiveThread(threadId);
      const data = await getThreadHistory(threadId, user.id);
      setMessages(data.history || []);
      setIsInitialChat(false);
      setInputMessage('');
    } catch (err:any) {
      console.error('Error loading thread');
      setErrorMessage(err.message|| 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !user || isBackendDown) return;
    setErrorMessage('');
    setIsLoading(true);
    const currentThreadId = activeThread;
    const message = inputMessage.trim();
    setInputMessage('');
  
    
    if (isInitialChat && !currentThreadId) {
      setIsInitialChat(false);
      setHasInitialInput(true);
    }
  
    
    setMessages(prev => [
      ...prev,
      { role: 'user', content: message, timestamp: new Date().toISOString() },
      { role: 'assistant', content: '', timestamp: new Date().toISOString(), isLoading: true }
    ]);
  
    try {
      const data = await sendMessageToBackend(message, currentThreadId!, user.id, isInitialChat);
  
      if (!activeThread && data.thread_id) {
        setActiveThread(data.thread_id);
      }
  
      
      setMessages(prev => {
        const lastUserIdx = prev.map(m => m.role).lastIndexOf('user');
        return [
          ...prev.slice(0, lastUserIdx + 1),
          { role: 'assistant', content: data.ai_message, timestamp: new Date().toISOString() }
        ];
      });
  
      if (isInitialChat && data.title) {
        setThreads(prev =>
          prev.map(thread =>
            thread.thread_id === data.thread_id ? { ...thread, title: data.title } : thread
          )
        );
      }
  
      await refreshThreads();
    } catch (err:any) {
      console.error('Error sending message');
      setErrorMessage(err.message|| 'Server error please try again later!');
    } finally {
      setIsLoading(false);
    }
  };
  

  const handleDownloadChat = async (threadId: string) => {
    if (!user) return;
    try {
      const data = await getThreadHistory(threadId, user.id);
      const thread = threads.find(t => t.thread_id === threadId);
      const title = thread?.title || "Chat History";
      generateChatHistoryPDF(data.history || [], title, user.firstName || user.username || 'User');
    } catch (error) {
      console.error('Error downloading chat');
    }
  };

  const toggleDarkMode = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    localStorage.setItem('darkMode', JSON.stringify(newMode));
    applyDarkMode(newMode);
  };

  const toggleLanguage = () => {
    const newLanguage = language === 'en' ? 'tr' : 'en';
    setLanguage(newLanguage);
    localStorage.setItem('language', newLanguage);
  };

  const toggleSettingsPanel = () => {
    setIsSettingsOpen(!isSettingsOpen);
  };

  return {
    t,
    user,
    isSidebarOpen,
    setIsSidebarOpen,
    isSettingsOpen,
    toggleSettingsPanel,
    isDarkMode,
    toggleDarkMode,
    language,
    toggleLanguage,
    threads,
    activeThread,
    messages,
    isLoading,
    inputMessage,
    setInputMessage,
    isInitialChat,
    hasInitialInput,
    handleCreateThread,
    handleDeleteThread,
    handleLoadThread,
    handleSendMessage,
    handleDownloadChat,
    errorMessage
  };
};