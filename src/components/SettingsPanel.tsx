interface SettingsPanelProps {
    isSettingsOpen: boolean;
    toggleSettingsPanel: () => void;
    isDarkMode: boolean;
    toggleDarkMode: (mode: boolean) => void;
    language: 'en' | 'tr';
    toggleLanguage: () => void;
    t: any;
  }
  
  function SettingsPanel({
    isSettingsOpen,
    toggleSettingsPanel,
    isDarkMode,
    toggleDarkMode,
    language,
    toggleLanguage,
    t,
  }: SettingsPanelProps) {
    return (
      <div className={`settings-panel ${isSettingsOpen ? 'open' : ''}`}>
        <div className="settings-header">
          <h2>{t.settings}</h2>
          <button className="close-settings" onClick={toggleSettingsPanel}>×</button>
        </div>
  
        <div className="settings-section">
          <h3>{t.theme}</h3>
          <div className="setting-item theme-buttons">
            <button
              className={`theme-option ${!isDarkMode ? 'active' : ''}`}
              onClick={() => toggleDarkMode(false)}
            >
              <span>🌞</span>
              <span>{t.lightMode}</span>
            </button>
            <button
              className={`theme-option ${isDarkMode ? 'active' : ''}`}
              onClick={() => toggleDarkMode(true)}
            >
              <span>🌙</span>
              <span>{t.darkMode}</span>
            </button>
          </div>
        </div>
  
        <div className="settings-section">
          <h3>{t.language}</h3>
          <div className="setting-item">
            <button
              className={`language-option ${language === 'en' ? 'active' : ''}`}
              onClick={toggleLanguage}
            >
              <img src="public/Images/GB.png" alt="English" />
              <span>{t.english}</span>
            </button>
            <button
              className={`language-option ${language === 'tr' ? 'active' : ''}`}
              onClick={toggleLanguage}
            >
              <img src="public/Images/TR.png" alt="Turkish" />
              <span>{t.turkish}</span>
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  export default SettingsPanel;
  