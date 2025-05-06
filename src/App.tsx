
import './App.css'
import { SignedIn, SignedOut } from "@clerk/clerk-react";
import PostSignInScreen from './components/PostSignInScreen';
import LandingPage from './components/LandingPage';

function App() {
  

  return (
    <div className="app-container">
      <SignedOut>
        <LandingPage />
      </SignedOut>
      <SignedIn>
        <PostSignInScreen/>
      </SignedIn>
    </div>
  )
}

export default App
