import { Message } from '../types/chatTypes';
import MessageBubble from './MessageBubble';
import { useEffect, useRef } from 'react';

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  inputMessage: string;
  setInputMessage: (message: string) => void;
  handleSendMessage: () => void;
  isInitialChat: boolean;
  hasInitialInput: boolean;
  t: any;
  user: any;
  errorMessage: string;
}

function ChatArea({
  messages,
  inputMessage,
  setInputMessage,
  handleSendMessage,
  isInitialChat,
  hasInitialInput,
  t,
  user,
  errorMessage,

}: ChatAreaProps) {
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="main-content">
      <div className="chat-container" ref={chatContainerRef}>
        {isInitialChat && !hasInitialInput && (
          <div className="initial-chat-message">
            <h2 className="welcome-animation">
              {t.welcome} {user?.firstName || user?.username}!
            </h2>
          </div>
        )}

        {errorMessage && (
          <div className="chat-error-box" style={{
            backgroundColor: '#ffe5e5',
            color: '#d8000c',
            padding: '12px',
            borderRadius: '8px',
            margin: '10px 0',
            textAlign: 'center',
            fontWeight: 'bold'
          }}>
            {errorMessage}
          </div>
        )}
        
        {messages.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))}
      </div>

      <div className="message-input-container">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={isInitialChat ? t.typeFirstMessage : t.typeMessage}
          className="chat-input"
        />
        <button
          onClick={handleSendMessage}
          disabled={!inputMessage.trim()}
          className="chat-button"
        >
          {t.send}
        </button>
      </div>
    </div>
  );
}

export default ChatArea;
