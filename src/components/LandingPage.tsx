import { SignInButton, SignUpButton } from "@clerk/clerk-react";
import { useState, useEffect } from 'react';
import './LandingPage.css';

const LandingPage = () => {
  
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('isDarkMode');
    return saved ? JSON.parse(saved) : false;
  });
  const [language, setLanguage] = useState<'en' | 'tr'>(() => {
    const saved = localStorage.getItem('language');
    return (saved === 'en' || saved === 'tr') ? saved : 'en';
  });

  
  useEffect(() => {
    localStorage.setItem('isDarkMode', JSON.stringify(isDarkMode));
    document.body.classList.toggle('dark-mode', isDarkMode);
  }, [isDarkMode]);

  useEffect(() => {
    localStorage.setItem('language', language);
  }, [language]);

  const toggleSettingsPanel = () => setIsSettingsOpen(!isSettingsOpen);
  const toggleDarkMode = (mode: boolean) => setIsDarkMode(mode);
  const toggleLanguage = () => setLanguage(prev => prev === 'en' ? 'tr' : 'en');

  
  const t = {
    settings: language === 'en' ? 'Settings' : 'Ayarlar',
    theme: language === 'en' ? 'Theme' : 'Tema',
    darkMode: language === 'en' ? 'Dark Mode' : 'Koyu Mod',
    lightMode: language === 'en' ? 'Light Mode' : 'Açık Mod',
    language: language === 'en' ? 'Language' : 'Dil',
    english: language === 'en' ? 'English' : 'İngilizce',
    turkish: language === 'en' ? 'Turkish' : 'Türkçe',
    signIn: language === 'en' ? 'Sign In' : 'Giriş Yap',
    signUp: language === 'en' ? 'Sign Up' : 'Kayıt Ol',
    description: language === 'en' 
      ? "An AI-powered assistant that offers clear, accessible financial legal guidance for everyone."
      : "Herkes için anlaşılır ve erişilebilir finansal hukuk rehberliği sunan yapay zekâ asistanı."
  };

  return (
    <div className={`landing-page${isDarkMode ? ' dark-mode' : ''}${language === 'en' ? ' lang-en' : ' lang-tr'}`}>
      <div className="top-bar">
        <div className="top-bar-logo">
          <img 
            src={isDarkMode ? "/Images/AI-ttorney Logo2White.png" : "/Images/AI-ttorney Logo2.png"} 
            alt="AI-ttorney Logo" 
            className="nav-logo" 
          />
        </div>
        <div className="top-bar-buttons">
          <div className="settings-container">
            <button className="settings-toggle" onClick={toggleSettingsPanel} title="Settings">
              ⚙
            </button>
            {/* Settings Panel */}
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
          </div>
          <SignInButton mode="modal">
            <button className="top-bar-button sign-in">{t.signIn}</button>
          </SignInButton>
          <SignUpButton mode="modal">
            <button className="top-bar-button sign-up">{t.signUp}</button>
          </SignUpButton>
        </div>
      </div>
      
      <div className="landing-container">
        <div className="landing-content">
          <h1 className="landing-title">
            {language === 'en' 
              ? "An AI-powered assistant that offers clear, accessible financial legal guidance"
              : "Herkes için anlaşılır ve erişilebilir finansal hukuk rehberliği sunan yapay zeka asistanı"
            }
          </h1>
          <p className="landing-subtitle">
            {language === 'en'
              ? "Get instant legal advice on Türkiye's financial law and personalized guidance for your legal matters."
              : "Anında Türkiye finansal hukuku üzerine danışmanlık ve kişiselleştirilmiş rehberlik alın."
            }
          </p>
        </div>
      </div>
    </div>
  );
};

export default LandingPage; 