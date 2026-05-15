// frontend/src/pages/Results.jsx
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getPipelineResult, getPipelineStatus } from '../api/pipeline'
import CleaningReport from '../components/pipeline/CleaningReport'
import EDAReport from '../components/pipeline/EDAReport'
import EvaluationReport from '../components/pipeline/EvaluationReport'
import ExplainabilityReport from '../components/pipeline/ExplainabilityReport'
import ModelsReport from '../components/pipeline/ModelsReport'
import PlanTimeline from '../components/pipeline/PlanTimeline'
import ProgressTracker from '../components/pipeline/ProgressTracker'
import DownloadPanel from '../components/reports/DownloadPanel'
import SkeletonLoader from '../components/shared/SkeletonLoader'

const tabs = ['Plan', 'Cleaning', 'EDA', 'Models', 'Explainability', 'Evaluation', 'Report']

export default function Results() {
  const { runId } = useParams()
  const [activeTab, setActiveTab] = useState('Plan')
  const statusQuery = useQuery({ queryKey: ['status', runId], queryFn: () => getPipelineStatus(runId), enabled: !!runId, refetchInterval: (query) => ['completed', 'failed'].includes(query.state.data?.status) ? false : 1500 })
  const resultQuery = useQuery({ queryKey: ['result', runId], queryFn: () => getPipelineResult(runId), enabled: !!runId && statusQuery.data?.status === 'completed' })
  const status = statusQuery.data?.status
  if (status === 'running' || status === 'not_found' || !status) return <ProgressTracker runId={runId} />
  if (status === 'failed') return <PipelineErrorCard errorData={statusQuery.data} onReset={() => window.history.back()} />
  if (resultQuery.isLoading) return <SkeletonLoader lines={8} height="h-12" />
  if (resultQuery.error) return <PipelineErrorCard errorData={{ error: resultQuery.error.message }} onReset={() => window.history.back()} />
  const result = resultQuery.data
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div><h2 className="text-2xl font-semibold text-slate-900">{result?.dataset_name}</h2><p className="text-sm text-slate-500">{result?.created_at ? new Date(result.created_at * 1000).toLocaleString() : ''}</p></div>
        <div className="flex items-center gap-3"><span className="rounded bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-700">{result?.status}</span><Link to={`/dashboard/chat/${runId}`} className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white">Chat With Data</Link></div>
      </div>
      <div className="overflow-hidden rounded-lg bg-white border border-slate-200">
        <div className="flex overflow-x-auto border-b border-slate-200">
          {tabs.map((tab) => <button key={tab} type="button" onClick={() => setActiveTab(tab)} className={`px-5 py-3 text-sm font-medium transition-all duration-200 ${activeTab === tab ? 'border-b-2 border-brand-500 text-slate-900' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'}`}>{tab}</button>)}
        </div>
        <div className="p-5 bg-white">
          {activeTab === 'Plan' ? <PlanTimeline planData={result?.plan_results} /> : null}
          {activeTab === 'Cleaning' ? <CleaningReport cleaningData={result?.cleaning_results} /> : null}
          {activeTab === 'EDA' ? <EDAReport edaData={result?.eda_results} /> : null}
          {activeTab === 'Models' ? <ModelsReport mlData={result?.ml_results} /> : null}
          {activeTab === 'Explainability' ? <ExplainabilityReport explainData={result?.explainability_results} /> : null}
          {activeTab === 'Evaluation' ? <EvaluationReport evaluationData={result?.evaluation_results} mlData={result?.ml_results} /> : null}
          {activeTab === 'Report' ? <DownloadPanel runId={runId} /> : null}
        </div>
      </div>
    </div>
  )
}

const getPipelineErrorSuggestion = (errorMsg = '') => {
  const msg = errorMsg.toLowerCase()

  if (msg.includes('no models trained') || msg.includes('all models failed')) {
    return {
      title: 'Model training failed',
      steps: [
        'Check your target column - it should have 2-20 unique values for classification',
        'Make sure your target column is not all nulls or a single constant value',
        "Try a different target column - look for columns like 'Survived', 'Churn', 'Price', 'Category'",
        'If using a regression target, ensure it has numeric values',
      ],
    }
  }
  if (msg.includes('less than 2 unique')) {
    return {
      title: 'Target column is constant',
      steps: [
        'The selected target column has only 1 unique value',
        'Pick a column that varies across rows',
      ],
    }
  }
  if (msg.includes('too few samples')) {
    return {
      title: 'Not enough data',
      steps: [
        'Your dataset has too few rows after cleaning',
        'Upload a dataset with at least 50 rows',
      ],
    }
  }
  return {
    title: 'Pipeline error',
    steps: ['Check the technical details below and try a different target column'],
  }
}

function PipelineErrorCard({ errorData, onReset }) {
  const errorMsg = errorData?.error || errorData?.message || 'Unknown error'
  const suggestion = getPipelineErrorSuggestion(errorMsg)

  return (
    <div style={{
      background: '#FEF2F2', border: '1px solid #FECACA',
      borderRadius: '10px', padding: '24px', maxWidth: '600px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
        <span style={{ fontSize: '20px', color: '#B91C1C' }}>!</span>
        <h3 style={{ color: '#B91C1C', margin: 0 }}>{suggestion.title}</h3>
      </div>

      <div style={{
        background: '#FFFBEB', border: '1px solid #FCD34D',
        borderRadius: '8px', padding: '14px', marginBottom: '16px',
      }}>
        <p style={{
          fontSize: '12px', fontWeight: 600, color: '#92400E',
          textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px',
        }}>
          What to do
        </p>
        <ul style={{ margin: 0, paddingLeft: '18px' }}>
          {suggestion.steps.map((step, i) => (
            <li key={i} style={{
              fontSize: '13px', color: '#78350F',
              marginBottom: '4px', lineHeight: '1.5',
            }}>
              {step}
            </li>
          ))}
        </ul>
      </div>

      <details style={{ marginBottom: '16px' }}>
        <summary style={{ fontSize: '12px', color: '#6B7280', cursor: 'pointer' }}>
          Technical details
        </summary>
        <pre style={{
          fontSize: '11px', background: '#F3F4F6', padding: '10px',
          borderRadius: '6px', marginTop: '8px', whiteSpace: 'pre-wrap',
          color: '#374151', maxHeight: '200px', overflowY: 'auto',
        }}>
          {JSON.stringify(errorData, null, 2)}
        </pre>
      </details>

      <button
        onClick={onReset}
        style={{
          background: '#4F46E5', color: 'white', border: 'none',
          borderRadius: '6px', padding: '10px 24px',
          fontSize: '14px', cursor: 'pointer', fontWeight: 500,
        }}
      >
        Try Again
      </button>
    </div>
  )
}
