// frontend/src/api/pipeline.js
import api from './axios'

export const runPipeline = (formData) =>
  api.post('/api/pipeline/run', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)

export const getPipelineStatus = (runId) =>
  api.get(`/api/pipeline/status/${runId}`).then((r) => r.data)

export const getPipelineResult = (runId) =>
  api.get(`/api/pipeline/result/${runId}`).then((r) => r.data)
