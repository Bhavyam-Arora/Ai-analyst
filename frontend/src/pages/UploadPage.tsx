import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import UploadZone from '../components/UploadZone'
import { uploadDocument } from '../services/api'
import { useDocumentStore } from '../store/useDocumentStore'

/**
 * UploadPage — Landing page for document upload.
 *
 * Flow:
 * 1. User selects a PDF or DOCX in the UploadZone
 * 2. On "Upload & Analyze", we POST to /api/upload
 * 3. On success, store docId + navigate to /analysis
 * 4. AnalysisPage handles triggering the LangGraph pipeline
 *
 * WHY SPLIT UPLOAD AND ANALYZE INTO TWO STEPS:
 * Upload is fast (seconds). Analysis is slow (30-60s).
 * By navigating immediately after upload, the user sees the analysis
 * page with a loading state rather than staring at the upload page.
 */
export default function UploadPage() {
  const navigate = useNavigate()
  const { setDocument, setIsAnalyzing, reset } = useDocumentStore()

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadDocument(file),
    onSuccess: (data) => {
      // Store document metadata, mark analysis as starting
      setDocument(data.doc_id, data.filename, data.page_count)
      setIsAnalyzing(true)
      // Navigate — AnalysisPage will kick off the analyze call
      navigate('/analysis')
    },
  })

  const handleFileAccepted = (file: File) => {
    reset() // Clear any previous document state
    uploadMutation.mutate(file)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-md">

        {/* Logo & header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-indigo-500/20 border border-indigo-500/30 mb-4">
            <span className="text-2xl">⚖️</span>
          </div>
          <h1 className="text-3xl font-bold text-white tracking-tight">AI Legal Analyst</h1>
          <p className="text-slate-400 mt-2 text-sm leading-relaxed max-w-xs mx-auto">
            Upload any legal document to extract key information,
            identify risks, and ask plain-English questions.
          </p>
        </div>

        {/* Upload card */}
        <div className="bg-white rounded-2xl p-7 shadow-2xl shadow-black/30">
          <h2 className="text-base font-semibold text-slate-800">Upload Document</h2>
          <p className="text-slate-500 text-sm mt-0.5 mb-5">
            Contracts, NDAs, lease deeds, service agreements
          </p>

          <UploadZone
            onFileAccepted={handleFileAccepted}
            isUploading={uploadMutation.isPending}
          />

          {uploadMutation.isError && (
            <p className="mt-4 text-red-500 text-sm text-center">
              Upload failed — please check the file and try again.
            </p>
          )}
        </div>

        {/* Feature pills */}
        <div className="mt-6 grid grid-cols-3 gap-3">
          {[
            { icon: '🔍', label: 'Extract', sub: 'Parties, dates, clauses' },
            { icon: '⚠️', label: 'Risk Analysis', sub: 'HIGH / MEDIUM / LOW' },
            { icon: '💬', label: 'Q&A', sub: 'Cited answers' },
          ].map(({ icon, label, sub }) => (
            <div
              key={label}
              className="bg-white/5 border border-white/10 rounded-xl p-3 text-center"
            >
              <span className="text-xl">{icon}</span>
              <p className="text-white text-xs font-semibold mt-1">{label}</p>
              <p className="text-slate-500 text-xs mt-0.5">{sub}</p>
            </div>
          ))}
        </div>

      </div>
    </div>
  )
}
