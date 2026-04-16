'use client'

import { useState, useRef, useEffect } from 'react'
import { useProLibraryStore } from '@/store'
import type { ChatMessage } from '@/store'

const SUGGESTED_QUESTIONS = [
  'What makes this swing so effective?',
  'What drills can help me copy this technique?',
  'How does the kinetic chain work here?',
]

export default function ProCoachChat() {
  const {
    selectedSwing,
    chatMessages,
    chatLoading,
    addChatMessage,
    appendToLastMessage,
    setChatLoading,
  } = useProLibraryStore()

  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Auto-scroll within the chat container on new messages (not the page)
  useEffect(() => {
    if (chatMessages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [chatMessages, chatLoading])

  const sendMessage = async (text: string) => {
    if (!text.trim() || !selectedSwing || chatLoading) return

    const userMsg: ChatMessage = { role: 'user', content: text.trim() }
    addChatMessage(userMsg)
    setInput('')

    const allMessages = [...chatMessages, userMsg]
    const assistantPlaceholder: ChatMessage = { role: 'assistant', content: '' }
    addChatMessage(assistantPlaceholder)
    setChatLoading(true)

    abortRef.current = new AbortController()

    try {
      const res = await fetch('/api/pro-coach', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          proSwingId: selectedSwing.id,
          messages: allMessages.map((m) => ({ role: m.role, content: m.content })),
        }),
        signal: abortRef.current.signal,
      })

      if (!res.ok || !res.body) {
        throw new Error('Chat request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        appendToLastMessage(decoder.decode(value, { stream: true }))
      }
      // Flush remaining bytes
      const remaining = decoder.decode()
      if (remaining) appendToLastMessage(remaining)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      appendToLastMessage('\n\n*Failed to get response. Please try again.*')
    } finally {
      setChatLoading(false)
      abortRef.current = null
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  if (!selectedSwing) return null

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden flex flex-col max-h-[500px]">
      {/* Header */}
      <div className="px-4 py-3 bg-white/5 border-b border-white/10 flex items-center gap-2">
        <span className="text-lg">🤖</span>
        <h3 className="text-sm font-semibold text-white">Pro Coach</h3>
        <span className="text-xs text-white/40">
          {selectedSwing.pros?.name ?? 'Pro'} · {selectedSwing.shot_type}
        </span>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-[120px]">
        {chatMessages.length === 0 && !chatLoading && (
          <div className="space-y-3">
            <p className="text-white/40 text-xs text-center">
              Ask anything about this swing technique
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="px-3 py-1.5 rounded-full text-xs bg-white/10 text-white/60 hover:bg-white/15 hover:text-white transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {chatMessages.map((msg, i) => (
          <div
            key={i}
            className={
              msg.role === 'user'
                ? 'bg-white/10 rounded-xl px-3 py-2 text-sm ml-8 text-white/80'
                : 'bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-3 py-2 text-sm mr-8'
            }
          >
            {msg.role === 'assistant' ? (
              <div className="space-y-1 text-white/80 text-sm leading-relaxed">
                <MarkdownText text={msg.content} />
                {chatLoading && i === chatMessages.length - 1 && (
                  <span className="inline-block w-1.5 h-4 bg-emerald-400 animate-pulse ml-0.5 rounded-sm" />
                )}
              </div>
            ) : (
              msg.content
            )}
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-white/10 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this swing..."
          disabled={chatLoading}
          className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-emerald-500/50 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || chatLoading}
          className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
            input.trim() && !chatLoading
              ? 'bg-emerald-500 hover:bg-emerald-400 text-white'
              : 'bg-white/10 text-white/30 cursor-not-allowed'
          }`}
        >
          Send
        </button>
      </form>
    </div>
  )
}

/** Minimal markdown renderer matching LLMCoachingPanel pattern. */
function MarkdownText({ text }: { text: string }) {
  const lines = text.split('\n')
  return (
    <>
      {lines.map((line, i) => {
        if (line.startsWith('## ')) {
          return (
            <h3 key={i} className="text-white font-bold text-base mt-3 mb-1">
              {line.replace('## ', '')}
            </h3>
          )
        }
        const parts = line.split(/(\*\*[^*]+\*\*)/)
        return (
          <p key={i} className={line.startsWith('-') ? 'pl-3' : ''}>
            {parts.map((part, j) =>
              part.startsWith('**') && part.endsWith('**') ? (
                <strong key={j} className="text-white">
                  {part.replace(/\*\*/g, '')}
                </strong>
              ) : (
                part
              )
            )}
          </p>
        )
      })}
    </>
  )
}
