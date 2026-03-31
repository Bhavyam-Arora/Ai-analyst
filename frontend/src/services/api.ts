/**
 * services/api.ts — Typed Axios client for the FastAPI backend
 *
 * All API calls go through this module. Components never call axios directly —
 * they call these typed functions. This means:
 * 1. A single place to update if endpoints or base URLs change
 * 2. TypeScript guarantees the request/response shapes match the backend
 * 3. Easy to mock in tests
 *
 * BASE URL:
 * In development, Vite proxies /api/* to http://localhost:8000, so we use
 * an empty baseURL. In production, set VITE_API_URL to your backend domain.
 */

import axios from 'axios'
import type { UploadResponse, AnalysisResponse, ChatResponse } from '../types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '',
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Upload a legal document (PDF or DOCX).
 * Returns a doc_id used for all subsequent operations on this document.
 */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const { data } = await api.post<UploadResponse>('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/**
 * Run the full LangGraph analysis pipeline on a previously uploaded document.
 * This can take 30-60 seconds — runs extraction, risk, and summary agents.
 */
export async function analyzeDocument(docId: string): Promise<AnalysisResponse> {
  const { data } = await api.post<AnalysisResponse>('/api/analyze', {
    doc_id: docId,
  })
  return data
}

/**
 * Ask a natural language question about a document.
 * Returns a grounded answer with page-level citations.
 */
export async function sendChatMessage(
  docId: string,
  question: string,
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/api/chat', {
    doc_id: docId,
    question,
  })
  return data
}
