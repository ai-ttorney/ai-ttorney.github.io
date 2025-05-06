import { Message } from '../types/chatTypes';
import LogoSpinner from './LogoSpinner';

interface MessageBubbleProps {
  message: Message;
}

function MessageBubble({ message }: MessageBubbleProps) {
  return (
    <div
      className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}${message.role === 'assistant' && message.isLoading ? ' no-bg' : ''}`}
    >
      <div className="message-content" style={{ whiteSpace: 'pre-line' }}>
        {message.isLoading ? (
          <LogoSpinner />
        ) : (
          message.content
        )}
      </div>
      <div className="message-timestamp">
        {new Date(message.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
}

export default MessageBubble;
