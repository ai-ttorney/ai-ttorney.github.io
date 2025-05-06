import { useEffect, useState } from 'react';

const images = [
  '/Images/logoanimation1.png',
  '/Images/logoanimation2.png'
];

const LogoSpinner = () => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % images.length);
    }, 300);
    return () => clearInterval(interval);
  }, []);

  return (
    <img
      src={images[index]}
      alt="Loading..."
      style={{ width: 30, height: 30, display: 'block', margin: '0 auto' }}
    />
  );
};

export default LogoSpinner; 