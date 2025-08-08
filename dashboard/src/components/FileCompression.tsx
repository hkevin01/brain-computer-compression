import React, { useRef, useState } from 'react'
import { CompressionResult, DecompressionResult } from '../types/api'
import PluginPicker from './PluginPicker'

export default function FileCompression() {
  const [selectedPlugin, setSelectedPlugin] = useState('')
  const [selectedMode, setSelectedMode] = useState('balanced')
  const [quality, setQuality] = useState(0.8)
  const [isProcessing, setIsProcessing] = useState(false)
  const [compressionResult, setCompressionResult] = useState<CompressionResult | null>(null)
  const [decompressionResult, setDecompressionResult] = useState<DecompressionResult | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [downloadProgress, setDownloadProgress] = useState(0)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const compressedFileRef = useRef<Blob | null>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.name.endsWith('.npy') || file.name.endsWith('.h5')) {
        setSelectedFile(file)
        setCompressionResult(null)
        setDecompressionResult(null)
      } else {
        alert('Please select a .npy or .h5 file')
        event.target.value = ''
      }
    }
  }

  const uploadFile = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const formData = new FormData()
      formData.append('file', file)

      const xhr = new XMLHttpRequest()

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100
          setUploadProgress(progress)
        }
      }

      xhr.onload = () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText)
          resolve(response.filename)
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`))
        }
      }

      xhr.onerror = () => reject(new Error('Upload failed'))

      xhr.open('POST', '/api/upload')
      xhr.send(formData)
    })
  }

  const compressFile = async () => {
    if (!selectedFile || !selectedPlugin) {
      alert('Please select a file and plugin')
      return
    }

    setIsProcessing(true)
    setUploadProgress(0)

    try {
      // Upload file first
      const filename = await uploadFile(selectedFile)

      // Compress
      const response = await fetch('/api/compress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename,
          plugin: selectedPlugin,
          mode: selectedMode,
          quality
        }),
      })

      if (!response.ok) {
        throw new Error(`Compression failed: ${response.statusText}`)
      }

      const result: CompressionResult = await response.json()
      setCompressionResult(result)

      // Get compressed file blob for download
      const downloadResponse = await fetch(`/api/download/${result.output_filename}`)
      if (downloadResponse.ok) {
        compressedFileRef.current = await downloadResponse.blob()
      }

    } catch (error) {
      console.error('Compression error:', error)
      alert(`Compression failed: ${error}`)
    } finally {
      setIsProcessing(false)
      setUploadProgress(0)
    }
  }

  const decompressFile = async () => {
    if (!compressionResult) {
      alert('No compressed file available')
      return
    }

    setIsProcessing(true)

    try {
      const response = await fetch('/api/decompress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: compressionResult.output_filename,
          plugin: selectedPlugin
        }),
      })

      if (!response.ok) {
        throw new Error(`Decompression failed: ${response.statusText}`)
      }

      const result: DecompressionResult = await response.json()
      setDecompressionResult(result)

    } catch (error) {
      console.error('Decompression error:', error)
      alert(`Decompression failed: ${error}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const downloadCompressed = () => {
    if (!compressedFileRef.current || !compressionResult) return

    const url = URL.createObjectURL(compressedFileRef.current)
    const a = document.createElement('a')
    a.href = url
    a.download = compressionResult.output_filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const downloadDecompressed = async () => {
    if (!decompressionResult) return

    setDownloadProgress(0)

    try {
      const response = await fetch(`/api/download/${decompressionResult.output_filename}`)
      if (!response.ok) throw new Error('Download failed')

      const reader = response.body?.getReader()
      const contentLength = +(response.headers.get('Content-Length') ?? 0)

      let receivedLength = 0
      const chunks = []

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          chunks.push(value)
          receivedLength += value.length

          if (contentLength > 0) {
            setDownloadProgress((receivedLength / contentLength) * 100)
          }
        }
      }

      const blob = new Blob(chunks)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = decompressionResult.output_filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

    } catch (error) {
      console.error('Download error:', error)
      alert(`Download failed: ${error}`)
    } finally {
      setDownloadProgress(0)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div>
      <div className="card">
        <div className="card-header">File Compression</div>
        <div className="card-content">
          <div className="row">
            <div className="col-md-6">
              <PluginPicker
                selectedPlugin={selectedPlugin}
                onPluginChange={setSelectedPlugin}
                selectedMode={selectedMode}
                onModeChange={setSelectedMode}
              />

              <div className="form-group">
                <label htmlFor="quality-slider">Quality Level: {quality.toFixed(2)}</label>
                <input
                  id="quality-slider"
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={quality}
                  onChange={(e) => setQuality(parseFloat(e.target.value))}
                  className="form-control"
                />
              </div>

              <div className="form-group">
                <label htmlFor="file-input">Select File (.npy or .h5)</label>
                <input
                  id="file-input"
                  type="file"
                  accept=".npy,.h5"
                  onChange={handleFileSelect}
                  ref={fileInputRef}
                  className="form-control"
                />
                {selectedFile && (
                  <small className="text-muted">
                    Selected: {selectedFile.name} ({formatFileSize(selectedFile.size)})
                  </small>
                )}
              </div>

              {uploadProgress > 0 && uploadProgress < 100 && (
                <div className="progress-container">
                  <label>Upload Progress</label>
                  <div className="progress">
                    <div
                      className="progress-bar"
                      style={{ width: `${uploadProgress}%` }}
                    >
                      {uploadProgress.toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              <div className="d-flex gap-2">
                <button
                  onClick={compressFile}
                  disabled={isProcessing || !selectedFile || !selectedPlugin}
                  className="btn btn-primary"
                >
                  {isProcessing ? 'Processing...' : 'Compress File'}
                </button>

                <button
                  onClick={decompressFile}
                  disabled={isProcessing || !compressionResult}
                  className="btn btn-secondary"
                >
                  {isProcessing ? 'Processing...' : 'Decompress'}
                </button>
              </div>
            </div>

            <div className="col-md-6">
              {compressionResult && (
                <div className="result-card">
                  <h4>Compression Results</h4>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-value">{compressionResult.compression_ratio.toFixed(2)}x</div>
                      <div className="metric-label">Compression Ratio</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value">{compressionResult.compression_time_ms.toFixed(1)}ms</div>
                      <div className="metric-label">Compression Time</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value">{formatFileSize(compressionResult.original_size)}</div>
                      <div className="metric-label">Original Size</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value">{formatFileSize(compressionResult.compressed_size)}</div>
                      <div className="metric-label">Compressed Size</div>
                    </div>
                    {compressionResult.snr_db && (
                      <div className="metric-card">
                        <div className="metric-value">{compressionResult.snr_db.toFixed(1)}dB</div>
                        <div className="metric-label">SNR</div>
                      </div>
                    )}
                    {compressionResult.psnr_db && (
                      <div className="metric-card">
                        <div className="metric-value">{compressionResult.psnr_db.toFixed(1)}dB</div>
                        <div className="metric-label">PSNR</div>
                      </div>
                    )}
                  </div>

                  <button
                    onClick={downloadCompressed}
                    className="btn btn-success mt-2"
                  >
                    Download Compressed File
                  </button>
                </div>
              )}

              {decompressionResult && (
                <div className="result-card mt-3">
                  <h4>Decompression Results</h4>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-value">{decompressionResult.decompression_time_ms.toFixed(1)}ms</div>
                      <div className="metric-label">Decompression Time</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value">{formatFileSize(decompressionResult.decompressed_size)}</div>
                      <div className="metric-label">Decompressed Size</div>
                    </div>
                    {decompressionResult.snr_db && (
                      <div className="metric-card">
                        <div className="metric-value">{decompressionResult.snr_db.toFixed(1)}dB</div>
                        <div className="metric-label">SNR</div>
                      </div>
                    )}
                    {decompressionResult.psnr_db && (
                      <div className="metric-card">
                        <div className="metric-value">{decompressionResult.psnr_db.toFixed(1)}dB</div>
                        <div className="metric-label">PSNR</div>
                      </div>
                    )}
                  </div>

                  {downloadProgress > 0 && downloadProgress < 100 && (
                    <div className="progress-container mt-2">
                      <label>Download Progress</label>
                      <div className="progress">
                        <div
                          className="progress-bar"
                          style={{ width: `${downloadProgress}%` }}
                        >
                          {downloadProgress.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  )}

                  <button
                    onClick={downloadDecompressed}
                    disabled={downloadProgress > 0 && downloadProgress < 100}
                    className="btn btn-success mt-2"
                  >
                    Download Decompressed File
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
