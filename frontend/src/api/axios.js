import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 120000,
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail = error?.response?.data?.detail
      || error?.response?.data?.message
      || error?.response?.data?.error
      || error?.message
      || 'Network error - is the backend running on port 8000?'
    console.error('[API Error]', detail)
    error.userMessage = detail
    return Promise.reject(error)
  },
)

export default api
