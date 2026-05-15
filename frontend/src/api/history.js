// frontend/src/api/history.js
import api from './axios'

export const getHistory = () => api.get('/api/history').then((r) => r.data)

export const deleteRun = (runId) =>
  api.delete(`/api/history/${runId}`).then((r) => r.data)
