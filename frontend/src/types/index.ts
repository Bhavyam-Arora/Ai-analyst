// TypeScript interfaces matching the backend Pydantic models exactly.
// Keep these in sync with backend/models/*.py

export interface UploadResponse {
  doc_id: string
  filename: string
  page_count: number
  chunk_count: number
  vector_count: number
  message: string
}

export interface ExtractedData {
  parties: string[] | null
  effective_date: string | null
  expiry_date: string | null
  payment_terms: string | null
  obligations: string[] | null
  termination_clauses: string[] | null
  jurisdiction: string | null
  governing_law: string | null
}

export interface RiskItem {
  severity: 'HIGH' | 'MEDIUM' | 'LOW'
  clause_text: string
  page_reference: number | null
  explanation: string
  recommendation: string
}

export interface AnalysisResponse {
  doc_id: string
  extracted_data: ExtractedData | null
  risks: RiskItem[] | null
  summary: string | null
  status: 'completed' | 'partial' | 'failed'
  error: string | null
}

export interface Citation {
  page_num: number
  chunk_text: string
  section_title: string | null
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  doc_id: string
}
