import type { Citation } from '../types'

interface CitationCardProps {
  citation: Citation
}

/**
 * CitationCard — Displays a single page citation from the Q&A agent.
 *
 * Shows the page number, optional section title, and the supporting excerpt.
 * In a future phase this can be wired to scroll/highlight the PDF viewer.
 */
export default function CitationCard({ citation }: CitationCardProps) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
      <div className="flex items-center gap-2 mb-2">
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full text-xs font-semibold">
          📖 Page {citation.page_num}
        </span>
        {citation.section_title && (
          <span className="text-slate-500 text-xs truncate">{citation.section_title}</span>
        )}
      </div>
      {citation.chunk_text && (
        <p className="text-slate-600 text-xs leading-relaxed line-clamp-4 italic">
          "{citation.chunk_text}"
        </p>
      )}
    </div>
  )
}
