//import { useState } from 'react'
import './App.css'
import { SignedIn, SignedOut, SignInButton } from "@clerk/clerk-react";
import PostSignInScreen from './PostSignInScreen';

function App() {
  //const [count, setCount] = useState(0)

  return (
    <header>
      <SignedOut>
        <SignInButton />
      </SignedOut>
      <SignedIn>
        <PostSignInScreen/>
      </SignedIn>
    </header>
  )
}

export default App
