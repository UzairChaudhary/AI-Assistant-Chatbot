'use client'
import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

export default function Home() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = { text: input, isUser: true }
    setMessages(prevMessages => [...prevMessages, userMessage])
    setInput('')

    try {
      const response = await axios.post('http://127.0.0.1:8000/chat', { message: input })
      const botMessage = { text: response.data.response, isUser: false }
      setMessages(prevMessages => [...prevMessages, botMessage])
    } catch (error) {
      console.error('Error sending message:', error)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="bg-green-500 text-white p-4">
        <h1 className="text-2xl font-bold">Uzair Bookstore AI Assistant</h1>
      </header>
      <main className="flex-grow overflow-y-auto p-4">
        {messages.map((message, index) => (
          <div key={index} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            <div className={`rounded-lg p-3 max-w-xs ${message.isUser ? 'bg-green-500 text-white' : 'bg-white'}`}>
              {message.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </main>
      <footer className="bg-white p-4">
        <form onSubmit={sendMessage} className="flex">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-grow rounded-l-lg p-2 border-t border-b border-l text-gray-800 border-gray-200 bg-white"
            placeholder="Type your message..."
          />
          <button type="submit" className="bg-green-500 text-white rounded-r-lg px-4 font-bold">Send</button>
        </form>
      </footer>
    </div>
  )
}