import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api requests to the FastAPI backend in development.
// This avoids needing CORS headers and keeps API calls relative (no hardcoded URLs).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
