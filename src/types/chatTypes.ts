export interface Thread {
    thread_id: string;
    title: string;
    created_at: string;
    last_updated: string;
    last_message?: string;
    is_initial: boolean;
  }
  
  export interface Message {
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    isLoading?: boolean;
  }
  
  export type Language = 'en' | 'tr';
  
  export type TranslationType = {
    welcome: string;
    newChat: string;
    settings: string;
    theme: string;
    language: string;
    darkMode: string;
    lightMode: string;
    english: string;
    turkish: string;
    typeFirstMessage: string;
    typeMessage: string;
    send: string;
    loading: string;
    deleteConfirm: string;
  };
  