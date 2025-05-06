import { useChat } from '../hooks/useChat';
import TopBar from './TopBar';
import SettingsPanel from './SettingsPanel';
import Sidebar from './Sidebar';
import ChatArea from './ChatArea';

function PostSignInScreen() {
  const chat = useChat();

  return (
    <div className={`app-container ${chat.isDarkMode ? 'dark-mode' : ''}`}>
      {/* Top Bar */}
      <TopBar
        isSidebarOpen={chat.isSidebarOpen}
        setIsSidebarOpen={chat.setIsSidebarOpen}
        toggleSettingsPanel={chat.toggleSettingsPanel}
        isDarkMode={chat.isDarkMode}
      />

      {/* Settings Panel */}
      <SettingsPanel
        isSettingsOpen={chat.isSettingsOpen}
        toggleSettingsPanel={chat.toggleSettingsPanel}
        isDarkMode={chat.isDarkMode}
        toggleDarkMode={chat.toggleDarkMode}
        language={chat.language}
        toggleLanguage={chat.toggleLanguage}
        t={chat.t}
      />

      {/* Sidebar */}
      <Sidebar
        threads={chat.threads}
        activeThread={chat.activeThread}
        handleCreateThread={chat.handleCreateThread}
        handleDeleteThread={chat.handleDeleteThread}
        handleLoadThread={chat.handleLoadThread}
        handleDownloadChat={chat.handleDownloadChat}
        t={chat.t}
      />

      {/* Main Chat Area */}
      <ChatArea
        messages={chat.messages}
        isLoading={chat.isLoading}
        inputMessage={chat.inputMessage}
        setInputMessage={chat.setInputMessage}
        handleSendMessage={chat.handleSendMessage}
        isInitialChat={chat.isInitialChat}
        hasInitialInput={chat.hasInitialInput}
        t={chat.t}
        user={chat.user}
        errorMessage={chat.errorMessage}

      />
    </div>
  );
}

export default PostSignInScreen;
