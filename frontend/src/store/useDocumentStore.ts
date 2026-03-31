/**
 * useDocumentStore — Global client state via Zustand
 *
 * WHY ZUSTAND OVER REACT CONTEXT:
 * Context re-renders every consumer on any state change. With Zustand,
 * components only re-render when the specific slice of state they subscribe
 * to changes. For a page with many components (ExtractedDataCard, RiskPanel,
 * ChatInterface) all reading different parts of the analysis result, this
 * avoids cascading re-renders.
 *
 * WHAT LIVES HERE (client state, not server state):
 * - The currently loaded document (docId, filename, pageCount)
 * - The analysis result for that document
 * - Whether analysis is currently running
 *
 * Server state (fetching, caching, loading) is handled by TanStack Query
 * in the page components that call the API.
 */

import { create } from 'zustand'
import type { AnalysisResponse } from '../types'

interface DocumentState {
  docId: string | null
  filename: string | null
  pageCount: number | null
  analysisResult: AnalysisResponse | null
  isAnalyzing: boolean

  setDocument: (docId: string, filename: string, pageCount: number) => void
  setAnalysisResult: (result: AnalysisResponse) => void
  setIsAnalyzing: (v: boolean) => void
  reset: () => void
}

const initialState = {
  docId: null,
  filename: null,
  pageCount: null,
  analysisResult: null,
  isAnalyzing: false,
}

export const useDocumentStore = create<DocumentState>((set) => ({
  ...initialState,

  setDocument: (docId, filename, pageCount) =>
    set({ docId, filename, pageCount }),

  setAnalysisResult: (result) =>
    set({ analysisResult: result, isAnalyzing: false }),

  setIsAnalyzing: (v) => set({ isAnalyzing: v }),

  // Called when user clicks "Upload New Document" — clears everything
  reset: () => set(initialState),
}))
