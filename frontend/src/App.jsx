import { BrowserRouter as Routerz_Hehe, Routes, Route, Link } from 'react-router-dom';
import { useState } from 'react'
import Home from './pages/Home';
import Navbar from './components/navbar';
import './App.css'
import Auth from './pages/Auth';

function App() {

  return (
    <>
      <Routerz_Hehe >

        <header className="">
          <Navbar />
        </header>
        <Routes className='mt-16'>
          <Route path='/' element={<Home />} />
          <Route path='/auth' element={<Auth />} />
          <Route path='*' element={<div className='text-center text-gray-600'>404</div>} />
        </Routes>

      </Routerz_Hehe>
    </>
  )
}

export default App
