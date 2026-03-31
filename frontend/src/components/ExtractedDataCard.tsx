import type { ExtractedData } from '../types'

interface ExtractedDataCardProps {
  data: ExtractedData
}

interface FieldRowProps {
  label: string
  value: string | string[] | null
}

function FieldRow({ label, value }: FieldRowProps) {
  if (value === null || (Array.isArray(value) && value.length === 0)) {
    return (
      <div className="py-3 border-b border-slate-100 last:border-0">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">{label}</span>
        <p className="text-slate-400 text-sm italic mt-1">Not specified in document</p>
      </div>
    )
  }

  return (
    <div className="py-3 border-b border-slate-100 last:border-0">
      <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">{label}</span>
      {Array.isArray(value) ? (
        <ul className="mt-1 space-y-1">
          {value.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
              <span className="text-indigo-400 mt-0.5 shrink-0">•</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-slate-800 text-sm mt-1">{value}</p>
      )}
    </div>
  )
}

/**
 * ExtractedDataCard — Renders the structured key information extracted by the extraction agent.
 *
 * Displays all fields from ExtractedData. Fields that are null are shown as
 * "Not specified" rather than hidden, so users can see what the AI looked for.
 */
export default function ExtractedDataCard({ data }: ExtractedDataCardProps) {
  return (
    <div className="divide-y divide-slate-100">
      <FieldRow label="Parties" value={data.parties} />
      <FieldRow label="Effective Date" value={data.effective_date} />
      <FieldRow label="Expiry Date" value={data.expiry_date} />
      <FieldRow label="Payment Terms" value={data.payment_terms} />
      <FieldRow label="Obligations" value={data.obligations} />
      <FieldRow label="Termination Clauses" value={data.termination_clauses} />
      <FieldRow label="Jurisdiction" value={data.jurisdiction} />
      <FieldRow label="Governing Law" value={data.governing_law} />
    </div>
  )
}
