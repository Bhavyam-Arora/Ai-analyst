import { useState } from 'react'
import type { AnalysisResponse } from '../types'
import ExtractedDataCard from './ExtractedDataCard'
import RiskPanel from './RiskPanel'

interface AnalysisDashboardProps {
  analysis: AnalysisResponse
}

type Tab = 'info' | 'risks' | 'summary'

/**
 * AnalysisDashboard — Tabbed view of the analysis pipeline output.
 *
 * Three tabs corresponding to the three LangGraph agents:
 * - "Key Info"  → extraction_agent output (ExtractedDataCard)
 * - "Risks"     → risk_agent output (RiskPanel)
 * - "Summary"   → summary_agent output (plain prose)
 */
export default function AnalysisDashboard({ analysis }: AnalysisDashboardProps) {
  const [activeTab, setActiveTab] = useState<Tab>('info')

  const riskCount = analysis.risks?.length ?? 0
  const highCount = analysis.risks?.filter((r) => r.severity === 'HIGH').length ?? 0

  const tabs: { id: Tab; label: string; badge?: string; badgeColor?: string }[] = [
    { id: 'info', label: 'Key Information' },
    {
      id: 'risks',
      label: 'Risks',
      badge: riskCount > 0 ? String(riskCount) : undefined,
      badgeColor: highCount > 0 ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700',
    },
    { id: 'summary', label: 'Summary' },
  ]

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex border-b border-slate-200 shrink-0">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={[
              'flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors',
              activeTab === tab.id
                ? 'border-indigo-600 text-indigo-700'
                : 'border-transparent text-slate-500 hover:text-slate-700',
            ].join(' ')}
          >
            {tab.label}
            {tab.badge && (
              <span className={`text-xs font-bold px-1.5 py-0.5 rounded-full ${tab.badgeColor}`}>
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-5">
        {activeTab === 'info' && (
          <>
            {analysis.extracted_data ? (
              <ExtractedDataCard data={analysis.extracted_data} />
            ) : (
              <EmptyState
                icon="🔍"
                title="Extraction failed"
                desc={analysis.error ?? 'Could not extract structured data from this document.'}
              />
            )}
          </>
        )}

        {activeTab === 'risks' && (
          <RiskPanel risks={analysis.risks ?? []} />
        )}

        {activeTab === 'summary' && (
          <>
            {analysis.summary ? (
              <div className="prose prose-slate max-w-none">
                <p className="text-slate-700 text-sm leading-relaxed whitespace-pre-wrap">
                  {analysis.summary}
                </p>
              </div>
            ) : (
              <EmptyState
                icon="📝"
                title="Summary unavailable"
                desc="The summary agent did not produce output for this document."
              />
            )}
          </>
        )}

        {/* Partial failure warning */}
        {analysis.status === 'partial' && analysis.error && (
          <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">
            <strong>Partial result:</strong> {analysis.error}
          </div>
        )}
      </div>
    </div>
  )
}

function EmptyState({ icon, title, desc }: { icon: string; title: string; desc: string }) {
  return (
    <div className="text-center py-12">
      <div className="text-4xl mb-3">{icon}</div>
      <p className="font-medium text-slate-700">{title}</p>
      <p className="text-sm text-slate-400 mt-1 max-w-sm mx-auto">{desc}</p>
    </div>
  )
}
