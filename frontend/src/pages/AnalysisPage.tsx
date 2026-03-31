import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { analyzeDocument } from '../services/api'
import { useDocumentStore } from '../store/useDocumentStore'
import AnalysisDashboard from '../components/AnalysisDashboard'
import ChatInterface from '../components/ChatInterface'

/**
 * AnalysisPage — Main view after a document is uploaded.
 *
 * Layout:
 * ┌─────────────┬───────────────────────────────┐
 * │  Left panel │  Right panel                  │
 * │  Doc info   │  AnalysisDashboard (tabs)     │
 * │  + status   ├───────────────────────────────┤
 * │             │  ChatInterface                 │
 * └─────────────┴───────────────────────────────┘
 *
 * State machine:
 * - No docId in store → redirect to / (user refreshed or landed directly)
 * - docId + isAnalyzing → trigger analyze mutation on mount → show loading
 * - docId + analysisResult → show dashboard
 */
export default function AnalysisPage() {
  const navigate = useNavigate()
  const {
    docId,
    filename,
    pageCount,
    analysisResult,
    isAnalyzing,
    setAnalysisResult,
    setIsAnalyzing,
  } = useDocumentStore()

  const analyzeMutation = useMutation({
    mutationFn: (id: string) => analyzeDocument(id),
    onSuccess: (data) => {
      setAnalysisResult(data)
    },
    onError: () => {
      setIsAnalyzing(false)
    },
  })

  // On mount: if we have a docId and analysis hasn't started yet, trigger it.
  // isAnalyzing is set to true by UploadPage right before navigating here.
  useEffect(() => {
    if (docId && isAnalyzing && !analysisResult && !analyzeMutation.isPending) {
      analyzeMutation.mutate(docId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId])

  // Guard: no document loaded → send back to upload
  if (!docId) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center gap-4">
        <p className="text-slate-600">No document loaded.</p>
        <button
          onClick={() => navigate('/')}
          className="px-5 py-2 bg-indigo-600 text-white rounded-lg text-sm font-semibold hover:bg-indigo-700"
        >
          Upload a Document
        </button>
      </div>
    )
  }

  const isLoading = isAnalyzing || analyzeMutation.isPending
  const hasError = analyzeMutation.isError

  return (
    <div className="h-screen bg-slate-50 flex flex-col overflow-hidden">

      {/* Top header bar */}
      <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-lg">⚖️</span>
          <span className="font-bold text-slate-800">AI Legal Analyst</span>
          {filename && (
            <>
              <span className="text-slate-300">/</span>
              <span className="text-slate-600 text-sm truncate max-w-xs" title={filename}>
                {filename}
              </span>
            </>
          )}
        </div>
        <button
          onClick={() => navigate('/')}
          className="text-sm text-indigo-600 hover:text-indigo-800 font-medium"
        >
          + New Document
        </button>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left panel — document metadata */}
        <aside className="w-56 shrink-0 bg-white border-r border-slate-200 flex flex-col p-5 gap-5">
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Document</p>
            <div className="flex items-start gap-2">
              <span className="text-2xl mt-0.5">📄</span>
              <div className="min-w-0">
                <p className="text-sm font-medium text-slate-800 truncate" title={filename ?? ''}>
                  {filename ?? '—'}
                </p>
                {pageCount !== null && (
                  <p className="text-xs text-slate-500 mt-0.5">{pageCount} pages</p>
                )}
              </div>
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Status</p>
            {isLoading && (
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-3 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm text-indigo-600">Analyzing…</span>
              </div>
            )}
            {!isLoading && hasError && (
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-red-500 rounded-full" />
                <span className="text-sm text-red-600">Analysis failed</span>
              </div>
            )}
            {!isLoading && analysisResult && (
              <>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full" />
                  <span className="text-sm text-emerald-700 font-medium">Analysis complete</span>
                </div>
                {analysisResult.status === 'partial' && (
                  <p className="text-xs text-amber-600 mt-1">Some results may be incomplete.</p>
                )}
              </>
            )}
          </div>

          {analysisResult?.risks && analysisResult.risks.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Risks</p>
              {(['HIGH', 'MEDIUM', 'LOW'] as const).map((sev) => {
                const count = analysisResult.risks!.filter((r) => r.severity === sev).length
                if (!count) return null
                const colors = {
                  HIGH: 'text-red-700 bg-red-50',
                  MEDIUM: 'text-amber-700 bg-amber-50',
                  LOW: 'text-emerald-700 bg-emerald-50',
                }
                return (
                  <div key={sev} className={`flex items-center justify-between px-2 py-1 rounded text-xs font-medium mb-1 ${colors[sev]}`}>
                    <span>{sev}</span>
                    <span className="font-bold">{count}</span>
                  </div>
                )
              })}
            </div>
          )}
        </aside>

        {/* Right panel — tabs + chat */}
        <main className="flex-1 flex flex-col overflow-hidden">

          {/* Loading state */}
          {isLoading && (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 text-slate-500">
              <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <div className="text-center">
                <p className="font-medium text-slate-700">Running analysis pipeline…</p>
                <p className="text-sm text-slate-400 mt-1">
                  Extracting key information · Identifying risks · Generating summary
                </p>
                <p className="text-xs text-slate-400 mt-1">(This takes 30–60 seconds)</p>
              </div>
            </div>
          )}

          {/* Error state */}
          {!isLoading && hasError && (
            <div className="flex-1 flex flex-col items-center justify-center gap-4">
              <span className="text-4xl">⚠️</span>
              <p className="text-slate-700 font-medium">Analysis failed</p>
              <p className="text-slate-500 text-sm">Check that the backend server is running.</p>
              <button
                onClick={() => {
                  setIsAnalyzing(true)
                  analyzeMutation.mutate(docId)
                }}
                className="px-5 py-2 bg-indigo-600 text-white rounded-lg text-sm font-semibold hover:bg-indigo-700"
              >
                Retry
              </button>
            </div>
          )}

          {/* Results */}
          {!isLoading && analysisResult && (
            <div className="flex-1 grid grid-rows-[1fr_320px] overflow-hidden">

              {/* Analysis dashboard (tabs) */}
              <div className="overflow-hidden border-b border-slate-200 bg-white">
                <AnalysisDashboard analysis={analysisResult} />
              </div>

              {/* Chat interface */}
              <div className="overflow-hidden bg-white">
                <ChatInterface docId={docId} />
              </div>

            </div>
          )}

        </main>
      </div>
    </div>
  )
}
