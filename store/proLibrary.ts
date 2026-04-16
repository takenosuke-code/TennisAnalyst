import { create } from 'zustand'
import type { Pro, ProSwing } from '@/lib/supabase'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ProLibraryStore {
  selectedPro: Pro | null
  selectedSwing: ProSwing | null
  loadingSwing: boolean
  chatMessages: ChatMessage[]
  chatLoading: boolean
  setSelectedPro: (pro: Pro | null) => void
  setSelectedSwing: (swing: ProSwing | null) => void
  setLoadingSwing: (v: boolean) => void
  addChatMessage: (msg: ChatMessage) => void
  appendToLastMessage: (chunk: string) => void
  setChatLoading: (v: boolean) => void
  resetChat: () => void
  resetAll: () => void
}

export const useProLibraryStore = create<ProLibraryStore>((set) => ({
  selectedPro: null,
  selectedSwing: null,
  loadingSwing: false,
  chatMessages: [],
  chatLoading: false,
  setSelectedPro: (selectedPro) => set({ selectedPro }),
  setSelectedSwing: (selectedSwing) => set({ selectedSwing }),
  setLoadingSwing: (loadingSwing) => set({ loadingSwing }),
  addChatMessage: (msg) =>
    set((state) => ({ chatMessages: [...state.chatMessages, msg] })),
  appendToLastMessage: (chunk) =>
    set((state) => {
      const msgs = [...state.chatMessages]
      if (msgs.length === 0) return state
      const last = msgs[msgs.length - 1]
      msgs[msgs.length - 1] = { ...last, content: last.content + chunk }
      return { chatMessages: msgs }
    }),
  setChatLoading: (chatLoading) => set({ chatLoading }),
  resetChat: () => set({ chatMessages: [], chatLoading: false }),
  resetAll: () =>
    set({
      selectedPro: null,
      selectedSwing: null,
      loadingSwing: false,
      chatMessages: [],
      chatLoading: false,
    }),
}))
