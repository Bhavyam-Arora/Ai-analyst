import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import type { FileRejection } from 'react-dropzone'

interface UploadZoneProps {
  onFileAccepted: (file: File) => void
  isUploading: boolean
}

const MAX_SIZE_MB = 20
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

export default function UploadZone({ onFileAccepted, isUploading }: UploadZoneProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [sizeError, setSizeError] = useState<string | null>(null)

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    setSizeError(null)
    if (rejectedFiles.length > 0) {
      const err = rejectedFiles[0].errors[0]
      if (err.code === 'file-too-large') {
        setSizeError(`File exceeds ${MAX_SIZE_MB}MB limit.`)
      } else if (err.code === 'file-invalid-type') {
        setSizeError('Only PDF and DOCX files are accepted.')
      }
      return
    }
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0])
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
    disabled: isUploading,
  })

  const handleUpload = () => {
    if (selectedFile && !isUploading) {
      onFileAccepted(selectedFile)
    }
  }

  const formatSize = (bytes: number) => (bytes / 1024 / 1024).toFixed(2)

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={[
          'border-2 border-dashed rounded-xl p-10 text-center transition-all select-none',
          isDragActive
            ? 'border-indigo-500 bg-indigo-50 scale-[1.01]'
            : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50',
          isUploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
        ].join(' ')}
      >
        <input {...getInputProps()} />
        <div className="text-4xl mb-3">📄</div>
        {isDragActive ? (
          <p className="text-indigo-600 font-semibold">Drop your document here</p>
        ) : (
          <div>
            <p className="text-slate-700 font-semibold">Drag &amp; drop your legal document</p>
            <p className="text-slate-500 text-sm mt-1">or click to browse files</p>
            <p className="text-slate-400 text-xs mt-3">PDF or DOCX &nbsp;·&nbsp; Max {MAX_SIZE_MB}MB</p>
          </div>
        )}
      </div>

      {/* Error */}
      {sizeError && (
        <p className="text-red-500 text-sm text-center">{sizeError}</p>
      )}

      {/* Selected file */}
      {selectedFile && !isUploading && (
        <div className="flex items-center justify-between px-4 py-3 bg-indigo-50 rounded-lg border border-indigo-200">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-indigo-500 shrink-0">📎</span>
            <span className="text-slate-800 text-sm font-medium truncate">{selectedFile.name}</span>
            <span className="text-slate-400 text-xs shrink-0">{formatSize(selectedFile.size)} MB</span>
          </div>
          <button
            onClick={() => setSelectedFile(null)}
            className="text-slate-400 hover:text-slate-600 text-xl leading-none ml-3 shrink-0"
            aria-label="Remove file"
          >
            ×
          </button>
        </div>
      )}

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!selectedFile || isUploading}
        className={[
          'w-full py-3 px-6 rounded-xl font-semibold text-sm transition-all',
          selectedFile && !isUploading
            ? 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-md hover:shadow-lg active:scale-[0.99]'
            : 'bg-slate-200 text-slate-400 cursor-not-allowed',
        ].join(' ')}
      >
        {isUploading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Uploading &amp; Analyzing…
          </span>
        ) : (
          'Upload & Analyze'
        )}
      </button>
    </div>
  )
}
