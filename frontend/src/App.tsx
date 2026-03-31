import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import UploadPage from './pages/UploadPage'
import AnalysisPage from './pages/AnalysisPage'

// QueryClient is created once and shared across all components via QueryClientProvider.
// It manages caching, deduplication, and background refetching for all server state.
const queryClient = new QueryClient({
  defaultOptions: {
    mutations: {
      // Don't retry failed mutations automatically — let the user decide to retry
      retry: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          {/* Redirect any unknown path to upload */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
