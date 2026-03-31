import type { RiskItem } from '../types'

interface RiskPanelProps {
  risks: RiskItem[]
}

const severityConfig = {
  HIGH: {
    bg: 'bg-red-50',
    border: 'border-red-200',
    badge: 'bg-red-100 text-red-700',
    dot: 'bg-red-500',
    label: 'HIGH',
  },
  MEDIUM: {
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    badge: 'bg-amber-100 text-amber-700',
    dot: 'bg-amber-500',
    label: 'MEDIUM',
  },
  LOW: {
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    badge: 'bg-emerald-100 text-emerald-700',
    dot: 'bg-emerald-500',
    label: 'LOW',
  },
}

function RiskCard({ risk }: { risk: RiskItem }) {
  const cfg = severityConfig[risk.severity] ?? severityConfig.LOW

  return (
    <div className={`rounded-xl border p-4 space-y-3 ${cfg.bg} ${cfg.border}`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${cfg.dot}`} />
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${cfg.badge}`}>
            {cfg.label}
          </span>
          {risk.page_reference !== null && (
            <span className="text-xs text-slate-500">Page {risk.page_reference}</span>
          )}
        </div>
      </div>

      {/* Clause text */}
      <blockquote className="text-sm text-slate-700 italic border-l-2 border-slate-300 pl-3 leading-relaxed">
        "{risk.clause_text}"
      </blockquote>

      {/* Explanation */}
      <div>
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Why it&apos;s a risk</p>
        <p className="text-sm text-slate-700 leading-relaxed">{risk.explanation}</p>
      </div>

      {/* Recommendation */}
      <div className="pt-1">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Recommendation</p>
        <p className="text-sm text-slate-700 leading-relaxed">{risk.recommendation}</p>
      </div>
    </div>
  )
}

/**
 * RiskPanel — Displays all identified risks sorted by severity (HIGH first).
 *
 * Each risk card shows the problematic clause text, why it's risky, and
 * what to do about it — matching the risk_agent output schema.
 */
export default function RiskPanel({ risks }: RiskPanelProps) {
  if (risks.length === 0) {
    return (
      <div className="text-center py-12 text-slate-500">
        <div className="text-4xl mb-3">✅</div>
        <p className="font-medium text-slate-700">No significant risks identified</p>
        <p className="text-sm mt-1 text-slate-400">The document appears to have standard clauses.</p>
      </div>
    )
  }

  // Sort: HIGH → MEDIUM → LOW
  const severityOrder = { HIGH: 0, MEDIUM: 1, LOW: 2 }
  const sorted = [...risks].sort(
    (a, b) => (severityOrder[a.severity] ?? 2) - (severityOrder[b.severity] ?? 2),
  )

  // Risk summary counts
  const counts = risks.reduce(
    (acc, r) => {
      acc[r.severity] = (acc[r.severity] ?? 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  return (
    <div className="space-y-4">
      {/* Summary badges */}
      <div className="flex flex-wrap gap-2 pb-2 border-b border-slate-100">
        {((['HIGH', 'MEDIUM', 'LOW'] as const)).map((sev) =>
          counts[sev] ? (
            <span
              key={sev}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${severityConfig[sev].badge}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${severityConfig[sev].dot}`} />
              {counts[sev]} {sev}
            </span>
          ) : null,
        )}
      </div>

      {/* Risk cards */}
      <div className="space-y-3">
        {sorted.map((risk, i) => (
          <RiskCard key={i} risk={risk} />
        ))}
      </div>
    </div>
  )
}
