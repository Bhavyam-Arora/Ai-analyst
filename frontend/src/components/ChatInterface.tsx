import { useState, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { sendChatMessage } from '../services/api'
import CitationCard from './CitationCard'
import type { Citation } from '../types'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  isError?: boolean
}

interface ChatInterfaceProps {
  docId: string
}

/**
 * ChatInterface — Conversational Q&A panel for the loaded document.
 *
 * Each question is sent to POST /api/chat, which retrieves relevant chunks
 * from Pinecone and returns a grounded answer with page citations.
 * The answer is strictly grounded — if the information isn't in the document,
 * the model says so rather than guessing.
 *
 * Messages are kept in local state (not in Zustand) because chat history
 * is ephemeral to the current session — no need to persist it globally.
 */
export default function ChatInterface({ docId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: "I've analyzed this document. Ask me anything — I'll answer from the document with page citations.",
    },
  ])
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const chatMutation = useMutation({
    mutationFn: ({ question }: { question: string }) =>
      sendChatMessage(docId, question),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          citations: data.citations,
        },
      ])
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Something went wrong. Please try your question again.',
          isError: true,
        },
      ])
    },
  })

  const handleSend = () => {
    const question = input.trim()
    if (!question || chatMutation.isPending) return

    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setInput('')
    chatMutation.mutate({ question })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-200 shrink-0">
        <h3 className="font-semibold text-slate-800 text-sm">Ask a Question</h3>
        <p className="text-xs text-slate-500 mt-0.5">Answers are grounded in the document with page citations</p>
      </div>

      {/* Message list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'user' ? (
              <div className="max-w-[80%] bg-indigo-600 text-white px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm leading-relaxed">
                {msg.content}
              </div>
            ) : (
              <div className="max-w-[90%] space-y-2">
                <div
                  className={[
                    'px-4 py-2.5 rounded-2xl rounded-tl-sm text-sm leading-relaxed',
                    msg.isError
                      ? 'bg-red-50 text-red-700 border border-red-200'
                      : 'bg-slate-100 text-slate-800',
                  ].join(' ')}
                >
                  {msg.content}
                </div>
                {msg.citations && msg.citations.length > 0 && (
                  <div className="space-y-1.5 pl-1">
                    {msg.citations.map((c, j) => (
                      <CitationCard key={j} citation={c} />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {chatMutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-slate-100 px-4 py-3 rounded-2xl rounded-tl-sm flex items-center gap-1.5">
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-200 shrink-0">
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about parties, clauses, obligations…"
            disabled={chatMutation.isPending}
            rows={2}
            className="flex-1 resize-none rounded-xl border border-slate-300 px-3 py-2 text-sm
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                       disabled:opacity-50 disabled:cursor-not-allowed placeholder:text-slate-400"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || chatMutation.isPending}
            className={[
              'px-4 py-2 rounded-xl font-semibold text-sm transition-all shrink-0',
              input.trim() && !chatMutation.isPending
                ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                : 'bg-slate-200 text-slate-400 cursor-not-allowed',
            ].join(' ')}
          >
            Send
          </button>
        </div>
        <p className="text-xs text-slate-400 mt-1.5">Press Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  )
}
