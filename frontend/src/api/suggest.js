// frontend/src/api/suggest.js
import api from './axios'

export const suggestTarget = (filePath) =>
  api.post('/api/suggest', { file_path: filePath }).then((r) => r.data)
