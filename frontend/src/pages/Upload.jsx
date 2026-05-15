import { useState } from 'react'
import { AlertTriangle, Check, Info, Loader2, Sparkles, Target, UploadCloud, XCircle, Zap } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import api from '../api/axios'
import { runPipeline } from '../api/pipeline'
import { suggestTarget } from '../api/suggest'
import usePipelineStore from '../store/pipelineStore'
import ErrorCard from '../components/shared/ErrorCard'

const models = [
  ['LogisticRegression', 'Fast linear classifier', 'Classification'],
  ['RandomForestClassifier', 'Robust tree ensemble', 'Classification'],
  ['XGBClassifier', 'Boosted classifier', 'Classification'],
  ['LinearRegression', 'Linear regression baseline', 'Regression'],
  ['RandomForestRegressor', 'Robust regression ensemble', 'Regression'],
  ['XGBRegressor', 'Boosted regressor', 'Regression'],
  ['GradientBoostingClassifier', 'Gradient boosted classifier', 'Classification'],
  ['Ridge', 'Regularized linear regression', 'Regression'],
  ['SVR', 'Support vector regression', 'Regression'],
]

function scoreColumnName(score) {
  return score?.column ?? score?.col
}

function scoreTaskType(score) {
  return score?.task_type ?? score?.task
}

function validateTargetColumn(colName, allScores) {
  if (!allScores || !colName) return null

  const colScore = allScores.find((s) => scoreColumnName(s) === colName)
  if (!colScore) return null

  if (colScore.score === -999) {
    return { type: 'error', msg: colScore.reason }
  }

  if (colScore.score < 10) {
    return {
      type: 'warning',
      msg: `Low suitability as target: ${colScore.reason}. Consider choosing a different column.`,
    }
  }

  if (scoreTaskType(colScore) === 'regression' && colScore.n_unique > 100) {
    return {
      type: 'info',
      msg: `Regression task detected - ${colScore.n_unique} unique values. Models will predict a continuous value.`,
    }
  }

  return null
}

export default function Upload() {
  const navigate = useNavigate()
  const { selectedModels, setSelectedModels, fastMode, setFastMode, setRunId } = usePipelineStore()
  const [file, setFile] = useState(null)
  const [filePath, setFilePath] = useState('')
  const [fileName, setFileName] = useState('')
  const [columns, setColumns] = useState([])
  const [previewRows, setPreviewRows] = useState([])
  const [goal, setGoal] = useState('')
  const [targetColumn, setTargetColumn] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [loading, setLoading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [targetWarning, setTargetWarning] = useState(null)
  const [autoDetecting, setAutoDetecting] = useState(false)
  const [suggestion, setSuggestion] = useState(null)

  const runAutoDetection = async (uploadedPath) => {
    if (!uploadedPath) return
    setAutoDetecting(true)
    setTargetColumn('')
    setGoal('')
    try {
      const data = await suggestTarget(uploadedPath)
      const scores = Array.isArray(data?.all_scores) ? data.all_scores : []
      const recommended = typeof data?.recommended_target === 'string' ? data.recommended_target : ''
      const suggestedGoal = typeof data?.suggested_goal === 'string' ? data.suggested_goal : ''

      setSuggestion(data)
      if (recommended) {
        setTargetColumn(recommended)
      }
      if (suggestedGoal) {
        setGoal(suggestedGoal)
      }
      setTargetWarning(null)

      const recScore = scores.find((s) => scoreColumnName(s) === recommended)
      const taskType = scoreTaskType(recScore)
      if (taskType) {
        const allowedTag = taskType === 'regression' ? 'Regression' : 'Classification'
        const compatibleModels = models.filter((m) => m[2] === allowedTag).map((m) => m[0])
        setSelectedModels(selectedModels.filter((m) => compatibleModels.includes(m)))
      }
    } catch (err) {
      console.warn('Auto-detection failed:', err)
    } finally {
      setAutoDetecting(false)
    }
  }

  const handleFileUpload = async (pickedFile) => {
    if (!pickedFile) return
    if (!pickedFile.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Please upload a .csv file only.')
      return
    }

    setFile(pickedFile)
    setUploading(true)
    setUploadError(null)
    setPreviewRows([])
    setColumns([])
    setFilePath('')
    setFileName('')
    setUploadSuccess(false)
    setTargetColumn('')
    setTargetWarning(null)
    setSuggestion(null)

    try {
      const formData = new FormData()
      formData.append('file', pickedFile)

      const res = await api.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      })

      const data = res?.data ?? {}
      const rows = Array.isArray(data.preview_rows) ? data.preview_rows : []
      const cols = Array.isArray(data.columns) ? data.columns : []
      const fpath = typeof data.file_path === 'string' ? data.file_path : ''
      const fname = typeof data.filename === 'string' ? data.filename : pickedFile.name

      if (cols.length === 0) {
        setUploadError('CSV appears to be empty or unreadable. Check the file and try again.')
        return
      }

      setPreviewRows(rows)
      setColumns(cols)
      setFilePath(fpath)
      setFileName(fname)
      setUploadSuccess(true)

      if (fpath) {
        runAutoDetection(fpath)
      }
    } catch (err) {
      const msg = err?.response?.data?.detail
        || err?.response?.data?.message
        || err?.userMessage
        || err?.message
        || 'Upload failed. Check that the backend is running on port 8000.'
      setUploadError(msg)
    } finally {
      setUploading(false)
    }
  }

  const onFile = (picked) => {
    handleFileUpload(picked?.[0])
  }

  const handleTargetChange = (colName) => {
    setTargetColumn(colName)
    setTargetWarning(validateTargetColumn(colName, suggestion?.all_scores))

    const score = suggestion?.all_scores?.find((s) => scoreColumnName(s) === colName)
    const taskType = scoreTaskType(score)
    if (taskType) {
      const allowedTag = taskType === 'regression' ? 'Regression' : 'Classification'
      const compatibleModels = models.filter((m) => m[2] === allowedTag).map((m) => m[0])
      setSelectedModels(selectedModels.filter((m) => compatibleModels.includes(m)))
    }
  }

  const toggleModel = (name) => {
    setSelectedModels(selectedModels.includes(name)
      ? selectedModels.filter((m) => m !== name)
      : [...selectedModels, name])
  }

  const canRunAnalysis = () => {
    if (!targetColumn || targetColumn === '' || targetColumn === 'Auto-detect') return false
    if (!columns.includes(targetColumn)) return false
    if (targetWarning?.type === 'error') return false
    return true
  }

  const submit = async () => {
    if (!targetColumn || targetColumn === '' || targetColumn === 'Auto-detect') {
      setUploadError('Please select a valid target column before running analysis.')
      return
    }

    if (!columns.includes(targetColumn)) {
      setUploadError(
        `Target column "${targetColumn}" not found in uploaded CSV. `
        + 'Please re-upload the file and select from the dropdown.',
      )
      return
    }

    if (!filePath) {
      setUploadError('No file uploaded. Please upload a CSV file first.')
      return
    }

    if (!goal || goal.trim().length < 5) {
      setUploadError('Please enter an analysis goal (at least a few words).')
      return
    }

    setLoading(true)
    setUploadError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('goal', goal.trim())
      form.append('target_col', targetColumn)
      form.append('selected_models', JSON.stringify(selectedModels))
      form.append('fast_mode', String(fastMode))
      form.append('run_explainability', 'true')
      const res = await runPipeline(form)
      setRunId(res.run_id)
      navigate(`/dashboard/results/${res.run_id}`)
    } catch (err) {
      setUploadError(err?.response?.data?.detail || err?.userMessage || err?.message)
    } finally {
      setLoading(false)
    }
  }

  const isTargetBlocked = targetWarning?.type === 'error'
  const currentTargetScore = suggestion?.all_scores?.find((s) => scoreColumnName(s) === targetColumn)
  const taskType = scoreTaskType(currentTargetScore)
  const displayModels = models.filter(([, , tag]) => {
    if (!taskType) return true
    if (taskType === 'regression') return tag === 'Regression'
    return tag === 'Classification'
  })

  return (
    <div className="space-y-6">
      <StepIndicator step={file ? 2 : 1} />
      {uploadError ? (
        <ErrorCard
          title="Upload failed"
          message="The analysis could not be started."
          detail={uploadError}
          onRetry={() => setUploadError(null)}
        />
      ) : null}
      <div
        onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files) }}
        onDragOver={(e) => e.preventDefault()}
        className="rounded-2xl border-2 border-dashed border-surface-500 p-12 text-center transition hover:border-brand-500 hover:bg-brand-500/5"
      >
        <UploadCloud className="mx-auto h-12 w-12 text-brand-300" />
        <p className="mt-4 text-lg font-semibold text-white">{fileName || file?.name || 'Drop CSV file here'}</p>
        {uploading ? <p className="mt-2 text-sm text-brand-200">Uploading and reading preview...</p> : null}
        {uploadSuccess ? <p className="mt-2 text-sm text-emerald-300">Upload ready</p> : null}
        <label className="mt-4 inline-flex cursor-pointer rounded-lg bg-brand-500 px-4 py-2 font-semibold text-white">
          <input type="file" accept=".csv" className="hidden" onChange={(e) => onFile(e.target.files)} />
          Select File
        </label>
      </div>

      {file ? (
        <div className="space-y-6">
          <PreviewTable columns={columns} rows={previewRows} />

          {autoDetecting && (
            <div className="auto-detect-banner auto-detect-loading">
              <Loader2 className="h-4 w-4 flex-shrink-0 animate-spin" />
              <span>Analyzing dataset for best target column...</span>
            </div>
          )}

          {suggestion && !autoDetecting && (
            <div className={`auto-detect-banner auto-detect-${suggestion.confidence}`}>
              <div className="banner-icon">
                {suggestion.confidence === 'high' ? <Check className="h-4 w-4" /> :
                  suggestion.confidence === 'medium' ? <AlertTriangle className="h-4 w-4" /> :
                    <Info className="h-4 w-4" />}
              </div>
              <div className="banner-text">
                <span className="banner-title">Auto-detected target:</span>{' '}
                <strong>{suggestion.recommended_target}</strong>
                <span className={`confidence-badge confidence-${suggestion.confidence}`}>
                  {suggestion.confidence} confidence
                </span>
                <span className="reason-text">{suggestion.reason}</span>
              </div>
            </div>
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-lg bg-surface-700 p-5">
              <label className="text-sm text-slate-300">Goal</label>
              <textarea
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                className="mt-2 h-28 w-full resize-none rounded-lg bg-surface-800 p-3 text-white outline-none ring-brand-500 focus:ring-2"
                placeholder="Describe your analysis goal..."
              />
              {suggestion && goal === suggestion.suggested_goal && (
                <span className="auto-generated-label">
                  <Sparkles className="inline h-3 w-3" /> Auto-generated - you can edit this
                </span>
              )}

              <div style={{ marginBottom: '16px' }}>
                <label style={{
                  display: 'block', fontSize: '13px',
                  fontWeight: 500, marginBottom: '6px', color: '#374151',
                }}>
                  Target Column
                  {autoDetecting && (
                    <span style={{ marginLeft: '8px', fontSize: '11px', color: '#6366F1' }}>
                      ⟳ Auto-detecting...
                    </span>
                  )}
                </label>

                <select
                  value={targetColumn}
                  onChange={(e) => handleTargetChange(e.target.value)}
                  disabled={autoDetecting}
                  style={{
                    width: '100%', padding: '10px 12px',
                    border: `1px solid ${targetWarning?.type === 'error' ? '#FCA5A5' : '#D1D5DB'}`,
                    borderRadius: '6px', fontSize: '14px',
                    background: autoDetecting ? '#F9FAFB' : 'white',
                    cursor: autoDetecting ? 'wait' : 'pointer',
                  }}
                >
                  {!targetColumn && (
                    <option value="" disabled>
                      {autoDetecting ? 'Detecting best column...' : 'Select target column...'}
                    </option>
                  )}

                  {columns.map((col) => {
                    const score = suggestion?.all_scores?.find((s) => s.col === col)
                    const isRecommended = col === suggestion?.recommended_target
                    const isBad = score?.score === -999

                    return (
                      <option key={col} value={col}>
                        {col}
                        {isRecommended ? ' ✓ recommended' : ''}
                        {isBad ? ' ✗ not suitable' : ''}
                      </option>
                    )
                  })}
                </select>

                {targetWarning && (
                  <div style={{
                    marginTop: '6px', padding: '8px 12px', borderRadius: '6px',
                    fontSize: '13px',
                    background: targetWarning.type === 'error' ? '#FEF2F2' : '#FFFBEB',
                    border: `1px solid ${targetWarning.type === 'error' ? '#FCA5A5' : '#FCD34D'}`,
                    color: targetWarning.type === 'error' ? '#B91C1C' : '#92400E',
                  }}>
                    {targetWarning.type === 'error' ? '⛔' : '⚠️'} {targetWarning.msg}
                  </div>
                )}

                {suggestion && !autoDetecting && targetColumn === suggestion.recommended_target && (
                  <div style={{
                    marginTop: '6px', padding: '8px 12px', borderRadius: '6px',
                    fontSize: '12px', background: '#ECFDF5',
                    border: '1px solid #6EE7B7', color: '#065F46',
                  }}>
                    ✓ Auto-detected ({suggestion.confidence} confidence) — {suggestion.reason}
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-lg bg-surface-700 p-5">
              <p className="text-sm text-slate-300">Mode</p>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <Mode selected={fastMode} onClick={() => setFastMode(true)} icon={Zap} title="Fast Mode" />
                <Mode selected={!fastMode} onClick={() => setFastMode(false)} icon={Target} title="Full Mode" />
              </div>
              <button
                type="button"
                disabled={!canRunAnalysis() || loading}
                onClick={submit}
                style={{
                  width: '100%', padding: '12px',
                  background: canRunAnalysis() && !loading ? '#4F46E5' : '#A5B4FC',
                  color: 'white', border: 'none', borderRadius: '8px',
                  fontSize: '15px', fontWeight: 500,
                  cursor: canRunAnalysis() && !loading ? 'pointer' : 'not-allowed',
                  transition: 'background 0.2s',
                }}
              >
                {loading ? 'Running Analysis...'
                  : !targetColumn ? 'Select a target column first'
                    : 'Run Analysis'}
              </button>
              {isTargetBlocked ? <p className="mt-2 text-center text-xs text-red-300">Select a valid target column to enable analysis.</p> : null}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            {displayModels.map(([name, desc, tag]) => (
              <button
                type="button"
                key={name}
                onClick={() => toggleModel(name)}
                className={`rounded-lg border p-4 text-left ${selectedModels.includes(name) ? 'border-brand-500 bg-brand-500/10' : 'border-surface-600 bg-surface-700'}`}
              >
                <div className="flex items-center gap-2 text-white">
                  <span className={`flex h-5 w-5 items-center justify-center rounded border ${selectedModels.includes(name) ? 'border-brand-500 bg-brand-500' : 'border-surface-500'}`}>
                    {selectedModels.includes(name) ? <Check className="h-3 w-3" /> : null}
                  </span>
                  {name}
                </div>
                <p className="mt-2 text-sm text-slate-400">{desc}</p>
                <p className="mt-2 text-xs text-brand-300">{tag}</p>
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}

function StepIndicator({ step }) {
  return (
    <div className="flex items-center justify-center gap-3">
      {[1, 2, 3].map((n) => (
        <div key={n} className="flex items-center gap-3">
          <div className={`flex h-9 w-9 items-center justify-center rounded-full font-semibold text-white ${n < step ? 'bg-emerald-500' : n === step ? 'bg-brand-500' : 'bg-surface-600'}`}>
            {n < step ? <Check className="h-4 w-4" /> : n}
          </div>
          {n < 3 ? <div className={`h-px w-16 ${n < step ? 'bg-emerald-500' : 'bg-surface-600'}`} /> : null}
        </div>
      ))}
    </div>
  )
}

function PreviewTable({ rows = [], columns = [] }) {
  if (!columns || columns.length === 0) return null
  if (!rows || rows.length === 0) {
    return <p style={{ color: '#6B7280', fontSize: '13px' }}>No preview available.</p>
  }

  return (
    <div style={{ overflowX: 'auto', maxHeight: '300px', overflowY: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '13px' }}>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={String(col)} style={{
                padding: '8px 12px', background: '#4F46E5', color: 'white',
                textAlign: 'left', whiteSpace: 'nowrap', position: 'sticky', top: 0,
              }}>
                {String(col)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} style={{ background: ri % 2 === 0 ? '#F9FAFB' : 'white' }}>
              {columns.map((col) => (
                <td key={String(col)} style={{
                  padding: '7px 12px', borderBottom: '1px solid #E5E7EB',
                  maxWidth: '200px', overflow: 'hidden',
                  textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {row?.[col] === null || row?.[col] === undefined
                    ? <span style={{ color: '#D1D5DB', fontStyle: 'italic' }}>null</span>
                    : String(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Mode({ selected, onClick, icon: Icon, title }) {
  return (
    <button type="button" onClick={onClick} className={`rounded-lg border p-4 text-left ${selected ? 'border-brand-500 bg-brand-500/10' : 'border-surface-600 bg-surface-800'}`}>
      <Icon className="h-5 w-5 text-brand-300" />
      <p className="mt-2 font-semibold text-white">{title}</p>
    </button>
  )
}
