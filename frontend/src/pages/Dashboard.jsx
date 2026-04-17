// frontend/src/pages/Dashboard.jsx
import { Link } from 'react-router-dom'
import { Activity, Calendar, Trophy } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { getHistory } from '../api/history'
import ErrorCard from '../components/shared/ErrorCard'
import MetricCard from '../components/shared/MetricCard'
import SkeletonLoader from '../components/shared/SkeletonLoader'

export default function Dashboard() {
  const { data: runs = [], isLoading, error } = useQuery({ queryKey: ['history'], queryFn: getHistory })
  if (isLoading) return <SkeletonLoader lines={8} height="h-12" />
  if (error) return <ErrorCard title="Dashboard unavailable" message="Could not load live run history." detail={error.message} />
  const scores = runs.map((r) => Number(r.score)).filter(Number.isFinite)
  const bestAccuracy = scores.length ? Math.max(...scores) : null
  const lastRun = runs[0]?.created_at ? new Date(runs[0].created_at * 1000).toLocaleString() : 'Never'
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Total Analyses" value={runs.length} icon={Activity} color="brand" />
        <MetricCard label="Best Accuracy" value={bestAccuracy === null ? 'N/A' : bestAccuracy.toFixed(4)} icon={Trophy} color="emerald" />
        <MetricCard label="Last Run" value={lastRun} icon={Calendar} color="blue" />
      </div>
      {!runs.length ? <div className="rounded-lg bg-surface-700 p-8 text-center text-white"><p>No analyses yet. Start your first.</p><Link to="/dashboard/upload" className="mt-4 inline-flex rounded-lg bg-brand-500 px-4 py-2 font-semibold">New Analysis</Link></div> : <RecentTable runs={runs.slice(0, 5)} />}
    </div>
  )
}

function RecentTable({ runs }) {
  return (
    <div className="overflow-hidden rounded-lg bg-surface-700">
      <h2 className="p-5 text-lg font-semibold text-white">Recent Analyses</h2>
      <table className="w-full text-left text-sm">
        <thead className="bg-surface-800 text-slate-300"><tr><th className="p-3">Dataset Name</th><th className="p-3">Date</th><th className="p-3">Status</th><th className="p-3">Best Model</th><th className="p-3">Score</th><th className="p-3">Action</th></tr></thead>
        <tbody>{runs.map((run) => <tr key={run.run_id} className="border-t border-surface-600"><td className="p-3 text-white">{run.dataset_name}</td><td className="p-3 text-slate-400">{new Date(run.created_at * 1000).toLocaleString()}</td><td className="p-3"><Status status={run.status} /></td><td className="p-3 text-slate-300">{run.best_model}</td><td className="p-3 text-slate-300">{typeof run.score === 'number' ? run.score.toFixed(4) : 'N/A'}</td><td className="p-3"><Link to={`/dashboard/results/${run.run_id}`} className="text-brand-300 hover:text-brand-100">View Results</Link></td></tr>)}</tbody>
      </table>
    </div>
  )
}

function Status({ status }) {
  const cls = status === 'completed' ? 'bg-emerald-500/20 text-emerald-200' : status === 'failed' ? 'bg-red-500/20 text-red-200' : 'bg-amber-500/20 text-amber-200'
  return <span className={`rounded px-2 py-1 text-xs ${cls}`}>{status}</span>
}

