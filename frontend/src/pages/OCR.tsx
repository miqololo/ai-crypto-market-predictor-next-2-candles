import { useState, useCallback } from 'react'
import { Upload, FileText, Loader2, CheckCircle2, XCircle, Image as ImageIcon, Copy, Download } from 'lucide-react'

const API_BASE = '/api'

interface OCRResult {
  text: string
  lines: string[]
  engine: string
  confidence: number | null
  word_count: number | null
  detected_languages?: string[]
  auto_detected?: boolean
  details?: Array<{
    text: string
    confidence: number | null
    bbox: number[][] | number[]
  }>
}

export default function OCR() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<OCRResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [languages, setLanguages] = useState('en')
  const [detail, setDetail] = useState(0)
  const [engine, setEngine] = useState<string>('')
  const [autoDetect, setAutoDetect] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  
  // Prompt-based extraction
  const [usePrompt, setUsePrompt] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [promptResult, setPromptResult] = useState<string | null>(null)
  const [promptProvider, setPromptProvider] = useState('local')  // Default to local for Armenian support
  const [promptLoading, setPromptLoading] = useState(false)

  const handleFileSelect = useCallback((selectedFile: File) => {
    // Validate file type
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select an image file (PNG, JPEG, GIF, BMP, TIFF, WebP)')
      return
    }

    // Validate file size (max 10MB)
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB')
      return
    }

    setFile(selectedFile)
    setError(null)
    setResult(null)

    // Create preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result as string)
    }
    reader.readAsDataURL(selectedFile)
  }, [])

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }, [handleFileSelect])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0])
    }
  }, [handleFileSelect])

  const handleExtract = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    // Use prompt-based extraction if enabled
    if (usePrompt && prompt.trim()) {
      await handlePromptExtract()
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setPromptResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const params = new URLSearchParams()
      if (!autoDetect && languages) params.append('languages', languages)
      if (detail !== 0) params.append('detail', detail.toString())
      if (engine) params.append('engine', engine)
      if (autoDetect) params.append('auto_detect', 'true')

      const response = await fetch(`${API_BASE}/ocr/extract?${params.toString()}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract text')
    } finally {
      setLoading(false)
    }
  }

  const handlePromptExtract = async () => {
    if (!file || !prompt.trim()) {
      setError('Please select a file and enter a prompt')
      return
    }

    setPromptLoading(true)
    setLoading(true)
    setError(null)
    setResult(null)
    setPromptResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const params = new URLSearchParams()
      params.append('prompt', prompt)
      params.append('provider', promptProvider)
      // For local provider, include languages for Armenian support
      if (promptProvider === 'local') {
        // Default to en,es,hy if no languages specified or empty
        const langsToUse = languages && languages.trim() ? languages : 'en,es,hy'
        params.append('languages', langsToUse)
      }

      const response = await fetch(`${API_BASE}/ocr/extract-with-prompt?${params.toString()}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setPromptResult(data.text)
      
      // Also set as regular result for display
      setResult({
        text: data.text,
        lines: data.text.split('\n').filter(l => l.trim()),
        engine: `${data.provider} (${data.model})`,
        confidence: null,
        word_count: data.text.split(/\s+/).length,
        detected_languages: null,
        auto_detected: false
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract with prompt')
    } finally {
      setPromptLoading(false)
      setLoading(false)
    }
  }

  const handleCopyText = () => {
    if (result?.text) {
      navigator.clipboard.writeText(result.text)
      // You could add a toast notification here
    }
  }

  const handleDownloadText = () => {
    if (result?.text) {
      const blob = new Blob([result.text], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `extracted_text_${Date.now()}.txt`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div className="min-h-screen bg-[#0f0f12] text-zinc-200 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <FileText className="w-8 h-8 text-emerald-500" />
            OCR Document Scanner
          </h1>
          <p className="text-zinc-400">
            Upload an image to extract text using OCR technology
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Side - Extracted Info */}
          <div className="space-y-4 order-2 lg:order-1">
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5 text-emerald-500" />
                Extracted Text
              </h2>

              {loading && (
                <div className="flex flex-col items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-emerald-500 mb-4" />
                  <p className="text-zinc-400">Extracting text...</p>
                </div>
              )}

              {error && (
                <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 mb-4">
                  <div className="flex items-center gap-2 text-red-400">
                    <XCircle className="w-5 h-5" />
                    <span className="font-medium">Error</span>
                  </div>
                  <p className="text-red-300 mt-2">{error}</p>
                </div>
              )}

              {result && !loading && (
                <div className="space-y-4">
                  {/* Result Header */}
                  <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
                    <div className="flex items-center gap-2 text-sm text-zinc-400">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      <span>Extracted using {result.engine}</span>
                      {result.auto_detected && result.detected_languages && (
                        <span className="ml-2">
                          • Detected: {result.detected_languages.join(', ')}
                        </span>
                      )}
                      {result.confidence !== null && (
                        <span className="ml-2">
                          • Confidence: {(result.confidence * 100).toFixed(1)}%
                        </span>
                      )}
                      {result.word_count !== null && (
                        <span className="ml-2">• {result.word_count} words</span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={handleCopyText}
                        className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm transition-colors flex items-center gap-2"
                        title="Copy to clipboard"
                      >
                        <Copy className="w-4 h-4" />
                        Copy
                      </button>
                      <button
                        onClick={handleDownloadText}
                        className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm transition-colors flex items-center gap-2"
                        title="Download as text file"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    </div>
                  </div>

                  {/* Extracted Text */}
                  <div className="bg-zinc-950/50 border border-zinc-800 rounded-lg p-4 max-h-[600px] overflow-y-auto">
                    <pre className="whitespace-pre-wrap font-mono text-sm text-zinc-200">
                      {result.text || 'No text extracted'}
                    </pre>
                  </div>

                  {/* Lines View */}
                  {result.lines && result.lines.length > 0 && (
                    <div className="mt-4">
                      <h3 className="text-sm font-medium text-zinc-400 mb-2">
                        Lines ({result.lines.length})
                      </h3>
                      <div className="bg-zinc-950/50 border border-zinc-800 rounded-lg p-4 max-h-[200px] overflow-y-auto">
                        {result.lines.map((line, idx) => (
                          <div key={idx} className="text-sm text-zinc-300 py-1 border-b border-zinc-800/50 last:border-0">
                            {line || <span className="text-zinc-500 italic">(empty line)</span>}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Detailed Results */}
                  {detail === 1 && result.details && result.details.length > 0 && (
                    <div className="mt-4">
                      <h3 className="text-sm font-medium text-zinc-400 mb-2">
                        Word Details ({result.details.length})
                      </h3>
                      <div className="bg-zinc-950/50 border border-zinc-800 rounded-lg p-4 max-h-[300px] overflow-y-auto">
                        <div className="space-y-2">
                          {result.details.slice(0, 20).map((detail, idx) => (
                            <div
                              key={idx}
                              className="text-xs bg-zinc-900/50 p-2 rounded border border-zinc-800/50"
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-mono text-zinc-200">{detail.text}</span>
                                {detail.confidence !== null && (
                                  <span className="text-zinc-400">
                                    {(detail.confidence * 100).toFixed(0)}%
                                  </span>
                                )}
                              </div>
                              {Array.isArray(detail.bbox) && detail.bbox.length > 0 && (
                                <div className="text-zinc-500 mt-1 text-xs">
                                  BBox: {JSON.stringify(detail.bbox)}
                                </div>
                              )}
                            </div>
                          ))}
                          {result.details.length > 20 && (
                            <p className="text-zinc-500 text-xs text-center py-2">
                              ... and {result.details.length - 20} more words
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {!result && !loading && !error && (
                <div className="text-center py-12 text-zinc-500">
                  <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Extracted text will appear here</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Side - File Uploader */}
          <div className="space-y-4 order-1 lg:order-2">
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-emerald-500" />
                Upload Document
              </h2>

              {/* File Upload Area */}
              <div
                className={`border-2 border-dashed rounded-lg p-8 transition-colors ${
                  dragActive
                    ? 'border-emerald-500 bg-emerald-500/10'
                    : 'border-zinc-700 hover:border-zinc-600'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  id="file-upload"
                  accept="image/*"
                  onChange={handleFileInput}
                  className="hidden"
                />
                <label
                  htmlFor="file-upload"
                  className="flex flex-col items-center justify-center cursor-pointer"
                >
                  {preview ? (
                    <div className="space-y-4 w-full">
                      <img
                        src={preview}
                        alt="Preview"
                        className="max-w-full max-h-64 mx-auto rounded-lg border border-zinc-700"
                      />
                      <div className="text-center">
                        <p className="text-sm text-zinc-300 mb-2">{file?.name}</p>
                        <p className="text-xs text-zinc-500">
                          {(file?.size || 0) / 1024} KB
                        </p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <ImageIcon className="w-12 h-12 text-zinc-500 mb-4" />
                      <p className="text-zinc-300 mb-2">
                        <span className="text-emerald-500 font-medium">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-sm text-zinc-500">
                        PNG, JPEG, GIF, BMP, TIFF, WebP (max 10MB)
                      </p>
                    </>
                  )}
                </label>
              </div>

              {/* File Info */}
              {file && (
                <div className="mt-4 p-3 bg-zinc-950/50 border border-zinc-800 rounded-lg">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-emerald-500" />
                      <span className="text-zinc-300">{file.name}</span>
                    </div>
                    <button
                      onClick={() => {
                        setFile(null)
                        setPreview(null)
                        setResult(null)
                        setError(null)
                      }}
                      className="text-zinc-500 hover:text-zinc-300"
                    >
                      <XCircle className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}

              {/* Options */}
              <div className="mt-6 space-y-4">
                <div>
                  <label className="flex items-center gap-2 mb-2">
                    <input
                      type="checkbox"
                      checked={usePrompt}
                      onChange={(e) => {
                        setUsePrompt(e.target.checked)
                        if (e.target.checked) {
                          setAutoDetect(false)
                        }
                      }}
                      className="w-4 h-4 rounded border-zinc-700 bg-zinc-900 text-emerald-600 focus:ring-emerald-500"
                    />
                    <span className="text-sm font-medium text-zinc-300">
                      Use AI Vision Model (Advanced)
                    </span>
                  </label>
                  <p className="text-xs text-zinc-500 ml-6">
                    Extract specific information using GPT-4 Vision, Claude, or Gemini with custom prompts
                  </p>
                </div>

                {usePrompt && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-zinc-300 mb-2">
                        AI Provider
                      </label>
                      <select
                        value={promptProvider}
                        onChange={(e) => setPromptProvider(e.target.value)}
                        className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      >
                        <option value="local">Local (OCR + LLM) - Supports Armenian</option>
                        <option value="openai">OpenAI (GPT-4 Vision)</option>
                        <option value="anthropic">Anthropic (Claude)</option>
                        <option value="google">Google (Gemini)</option>
                        <option value="ollama">Ollama (LLaVA - Local Vision)</option>
                      </select>
                      <p className="text-xs text-zinc-500 mt-1">
                        {promptProvider === 'local' 
                          ? 'Uses Tesseract OCR (Armenian support) + your local LLM. Works offline.'
                          : promptProvider === 'ollama'
                          ? 'Direct vision model. Limited Armenian support.'
                          : 'Cloud-based vision models'}
                      </p>
                    </div>

                    {promptProvider === 'local' && (
                      <div>
                        <label className="block text-sm font-medium text-zinc-300 mb-2">
                          Languages (comma-separated)
                        </label>
                        <input
                          type="text"
                          value={languages || 'en,es,hy'}
                          onChange={(e) => setLanguages(e.target.value)}
                          placeholder="en, es, hy"
                          className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                        />
                        <p className="text-xs text-zinc-500 mt-1">
                          Language codes for OCR: en (English), es (Spanish), hy (Armenian), etc.
                          Default: en,es,hy (includes Armenian support). Leave empty to use default.
                        </p>
                      </div>
                    )}

                    <div>
                      <label className="block text-sm font-medium text-zinc-300 mb-2">
                        Prompt (What to extract?)
                      </label>
                      <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="e.g., Extract all dates and amounts from this invoice"
                        rows={3}
                        className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
                      />
                      <p className="text-xs text-zinc-500 mt-1">
                        Describe exactly what information you want to extract from the document
                      </p>
                    </div>
                  </>
                )}

                {!usePrompt && (
                  <div>
                    <label className="flex items-center gap-2 mb-2">
                      <input
                        type="checkbox"
                        checked={autoDetect}
                        onChange={(e) => {
                          setAutoDetect(e.target.checked)
                          if (e.target.checked) {
                            setLanguages('')
                          }
                        }}
                        className="w-4 h-4 rounded border-zinc-700 bg-zinc-900 text-emerald-600 focus:ring-emerald-500"
                      />
                      <span className="text-sm font-medium text-zinc-300">
                        Auto-detect languages
                      </span>
                    </label>
                    <p className="text-xs text-zinc-500 ml-6">
                      Automatically detect between: English, Spanish, and Armenian
                    </p>
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-zinc-300 mb-2">
                    OCR Engine
                  </label>
                  <select
                    value={engine}
                    onChange={(e) => setEngine(e.target.value)}
                    className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  >
                    <option value="">Auto (default)</option>
                    <option value="tesseract">Tesseract (Recommended on macOS ARM)</option>
                    <option value="easyocr">EasyOCR</option>
                  </select>
                  <p className="text-xs text-zinc-500 mt-1">
                    {engine === 'tesseract' || !engine ? (
                      <>Tesseract is recommended on macOS ARM to avoid threading issues</>
                    ) : (
                      <>EasyOCR supports more languages but may crash on macOS ARM</>
                    )}
                  </p>
                </div>

                {!usePrompt && !autoDetect && (
                  <div>
                    <label className="block text-sm font-medium text-zinc-300 mb-2">
                      Languages (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={languages}
                      onChange={(e) => setLanguages(e.target.value)}
                      placeholder="en, es, fr"
                      disabled={autoDetect}
                      className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
                    />
                    <p className="text-xs text-zinc-500 mt-1">
                      Language codes: en (English), es (Spanish), fr (French), hy (Armenian), etc.
                      {engine === 'tesseract' && ' Use "eng" for English, "hye" for Armenian with Tesseract'}
                    </p>
                  </div>
                )}

                {!usePrompt && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-zinc-300 mb-2">
                        OCR Engine
                      </label>
                      <select
                        value={engine}
                        onChange={(e) => setEngine(e.target.value)}
                        className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      >
                        <option value="">Auto (default)</option>
                        <option value="tesseract">Tesseract (Recommended on macOS ARM)</option>
                        <option value="easyocr">EasyOCR</option>
                      </select>
                      <p className="text-xs text-zinc-500 mt-1">
                        {engine === 'tesseract' || !engine ? (
                          <>Tesseract is recommended on macOS ARM to avoid threading issues</>
                        ) : (
                          <>EasyOCR supports more languages but may crash on macOS ARM</>
                        )}
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-zinc-300 mb-2">
                        Detail Level
                      </label>
                      <select
                        value={detail}
                        onChange={(e) => setDetail(Number(e.target.value))}
                        className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      >
                        <option value={0}>Text Only</option>
                        <option value={1}>Text with Bounding Boxes</option>
                      </select>
                      <p className="text-xs text-zinc-500 mt-1">
                        Level 1 includes word positions and confidence scores
                      </p>
                    </div>
                  </>
                )}

                <div>
                  <label className="block text-sm font-medium text-zinc-300 mb-2">
                    Detail Level
                  </label>
                  <select
                    value={detail}
                    onChange={(e) => setDetail(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  >
                    <option value={0}>Text Only</option>
                    <option value={1}>Text with Bounding Boxes</option>
                  </select>
                  <p className="text-xs text-zinc-500 mt-1">
                    Level 1 includes word positions and confidence scores
                  </p>
                </div>
              </div>

              {/* Extract Button */}
              <button
                onClick={handleExtract}
                disabled={!file || loading || (usePrompt && !prompt.trim())}
                className={`w-full mt-6 px-4 py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 ${
                  file && !loading && (!usePrompt || prompt.trim())
                    ? 'bg-emerald-600 hover:bg-emerald-700 text-white'
                    : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                }`}
              >
                {loading || promptLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    {usePrompt ? 'Analyzing with AI...' : 'Extracting...'}
                  </>
                ) : (
                  <>
                    <FileText className="w-5 h-5" />
                    {usePrompt ? 'Extract with AI Prompt' : 'Extract Text'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
