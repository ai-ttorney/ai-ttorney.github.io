import { UserButton } from "@clerk/clerk-react";

interface TopBarProps {
    isSidebarOpen: boolean;
    setIsSidebarOpen: (open: boolean) => void;
    toggleSettingsPanel: () => void;
    isDarkMode: boolean;
  }
  
  function TopBar({setIsSidebarOpen, toggleSettingsPanel, isDarkMode }: TopBarProps) {
    return (
      <div className="top-bar">
        <div className="top-bar-left">
          <button
            className="sidebar-toggle"
            onMouseEnter={() => setIsSidebarOpen(true)}
          >
            ☰
          </button>
          <div className="top-bar-logo">
            <img 
              src={isDarkMode ? "/Images/AI-ttorney Logo2White.png" : "/Images/AI-ttorney Logo2.png"} 
              alt="AI-ttorney Logo" 
              className="nav-logo" 
            />
          </div>
        </div>
        <div className="top-bar-buttons">
          <button className="settings-toggle" onClick={toggleSettingsPanel} title="Settings">
            ⚙
          </button>
          <div className="user-button-container">
            <div style={{ width: "40px", height: "40px" }}>
             
              <UserButton />
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  export default TopBar;
  