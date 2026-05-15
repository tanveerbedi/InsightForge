// frontend/src/components/shared/ErrorCard.jsx
import { useState } from 'react'
import { AlertCircle, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'

function getSuggestion(msg) {
  if (!msg) return 'Check that your target column contains meaningful, non-constant values suitable for prediction.'
  const m = msg.toLowerCase()
  if (m.includes('less than 2 unique') || m.includes('only 1 unique') || m.includes('cannot be used for modeling'))
    return 'The selected target column has only one unique value (constant column). Choose a different target column — ideally one with multiple distinct values like a category, score, or amount.'
  if (m.includes('unique identifier') || m.includes('every row is different'))
    return 'The selected target column appears to be an ID column where every row is different. Pick a column that represents a category or measurable outcome.'
  if (m.includes('not found') || m.includes('does not exist'))
    return "The target column name doesn't match any column in your CSV. Re-upload the file and reselect the target column from the dropdown."
  if (m.includes('missing values') || m.includes('too many nulls'))
    return 'The target column has too many null/empty values. Try a different target column or clean the data first.'
  if (m.includes('at least 50 rows'))
    return 'Your dataset is too small. Upload a CSV with at least 50 rows of data.'
  if (m.includes('at least 2 columns'))
    return 'Your dataset needs at least 2 columns (features + target). Check that the CSV is formatted correctly.'
  return 'Check that your target column contains meaningful, non-constant values suitable for prediction.'
}

function parseErrorDetail(detail) {
  if (!detail) return { message: null, data: null }
  if (typeof detail === 'string') {
    try {
      const parsed = JSON.parse(detail)
      return { message: parsed?.error || parsed?.detail || detail, data: parsed }
    } catch {
      return { message: detail, data: null }
    }
  }
  if (typeof detail === 'object') {
    return { message: detail.error || detail.detail || JSON.stringify(detail), data: detail }
  }
  return { message: String(detail), data: null }
}

export default function ErrorCard({ title = 'Something went wrong', message = 'The request could not be completed.', detail, onRetry }) {
  const [showDetail, setShowDetail] = useState(false)
  const { message: errorMessage, data: errorData } = parseErrorDetail(detail)
  const suggestion = getSuggestion(errorMessage)
  const hasRawDetail = detail && typeof detail === 'string' && detail.length > 100

  return (
    <div className="rounded-xl border border-red-200 bg-red-50 p-6 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-red-100">
          <AlertCircle className="h-5 w-5 text-red-600" />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-lg font-semibold text-red-900">{title}</h3>
          <p className="mt-1 text-sm text-red-700">{message}</p>

          {/* Error reason */}
          {errorMessage ? (
            <div className="mt-4 rounded-lg border border-red-200 bg-white p-4">
              <p className="text-sm font-medium text-red-800">{errorMessage}</p>
            </div>
          ) : null}

          {/* Model Failures & Diagnostics */}
          {errorData?.model_failures?.length > 0 && (
            <div className="mt-4 rounded-lg border border-red-200 bg-white p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-red-800 mb-2">Model Failures</p>
              <ul className="space-y-2">
                {errorData.model_failures.map((mf, idx) => (
                  <li key={idx} className="text-xs text-slate-700 bg-slate-50 p-2 rounded">
                    <strong>{mf.name}:</strong> <span className="text-red-600 break-words">{String(mf.error)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {errorData?.diagnostics && (
            <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-blue-800 mb-2">Pipeline Diagnostics</p>
              <div className="grid grid-cols-2 gap-2 text-xs text-blue-900">
                <div><strong>Task Type:</strong> {errorData.diagnostics.task_type}</div>
                <div><strong>Rows Evaluated:</strong> {errorData.diagnostics.n_rows}</div>
                <div><strong>Features Count:</strong> {errorData.diagnostics.n_features}</div>
                <div><strong>Target Unique Values:</strong> {errorData.diagnostics.target_unique_values}</div>
              </div>
            </div>
          )}

          {/* Actionable suggestion */}
          <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">What to do</p>
            <p className="mt-1 text-sm text-amber-800">{suggestion}</p>
          </div>

          {/* Raw detail toggle */}
          {hasRawDetail ? (
            <div className="mt-3">
              <button
                type="button"
                onClick={() => setShowDetail((v) => !v)}
                className="inline-flex items-center gap-1 text-xs font-medium text-red-500 hover:text-red-700 transition-colors"
              >
                Technical Details
                {showDetail ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              </button>
              {showDetail ? (
                <pre className="mt-2 max-h-48 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-300 leading-relaxed">{String(detail)}</pre>
              ) : null}
            </div>
          ) : null}

          {/* Retry button */}
          {onRetry ? (
            <button
              type="button"
              onClick={onRetry}
              className="mt-4 inline-flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
              Try Again
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}
