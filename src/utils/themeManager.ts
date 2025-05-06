export const applyDarkMode = (isDark: boolean) => {
    if (isDark) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
  };
  
  export const loadDynamicResources = () => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/styles.css';
    document.head.appendChild(link);
  
    const script = document.createElement('script');
    script.src = '/script.js';
    script.defer = true;
    document.body.appendChild(script);
  
    return () => {
      document.head.removeChild(link);
      document.body.removeChild(script);
    };
  };
  