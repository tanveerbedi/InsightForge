// frontend/src/api/chat.js
import api from './axios'

export const sendMessage = (runId, question, history) =>
  api.post(`/api/chat/${runId}`, { question, history }).then((r) => r.data)
