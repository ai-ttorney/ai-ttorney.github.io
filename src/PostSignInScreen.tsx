import { useEffect } from 'react';
import { UserButton } from "@clerk/clerk-react";


function PostSignInScreen() {
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
    <div>
      {/* Sidebar with Logo, Settings, and Old Conversations */}
      <div className="sidebar" id="sidebar">
        <div className="logo">
          <img src="Images/AITTORNEY logo.png" alt="AITTORNEY Logo" />
        </div>

        <div className="profile">
          <UserButton/>
        </div>

        <div className="conversations">
          <h3>Old Conversations</h3>
          <ul>
            <li>Conversation 1</li>
            <li>Conversation 2</li>
            <li>Conversation 3</li>
          </ul>
        </div>
      </div>

      {/* Sidebar Toggle Button */}
      <button id="sidebarToggle" className="sidebar-toggle">
        ☰
      </button>

      {/* Main Chat Area */}
      <div className="main-container">
        <div className="chat-container" id="chatContainer"></div>
        <div className="message-input">
          <input type="text" id="message" placeholder="Type your message here..." />
          <button id="sendBtn">Send</button>
        </div>
      </div>
    </div>
  );
}

export default PostSignInScreen;
