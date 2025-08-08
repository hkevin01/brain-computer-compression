# Brain-Computer Interface Data Compression GUI

A production-ready web dashboard for real-time monitoring and control of neural data compression algorithms.

## Features

### 🚀 Real-Time Live Stream
- Live compression monitoring with WebSocket streaming
- Real-time charts showing compression ratio, latency, and signal quality metrics
- Plugin selection with mode and quality controls
- Per-channel analysis and visualization

### 📁 File Compression
- Upload and compress neural data files (.npy, .h5 formats)
- Download compressed and decompressed files
- Detailed compression metrics and quality analysis
- Progress indicators for upload/download operations

### 📊 Benchmark Suite
- Compare multiple compression plugins side-by-side
- Automated benchmarking with configurable test parameters
- Interactive charts showing performance trade-offs
- Quality vs latency analysis with scatter plots

## Quick Start

### Prerequisites
- Python 3.8+ with bci_compression package installed
- Node.js 16+ for frontend development

### Backend Setup

1. **Install Dependencies**
   ```bash
   pip install fastapi uvicorn websockets python-multipart
   ```

2. **Start the Server**
   ```bash
   cd brain-computer-compression
   python scripts/telemetry_server.py
   ```
   Server runs on `http://localhost:8000`

### Frontend Setup

1. **Install Dependencies**
   ```bash
   cd dashboard
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```
   Dashboard available at `http://localhost:3000`

3. **Build for Production**
   ```bash
   npm run build
   npm run preview
   ```

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/api/plugins` | GET | List available compression plugins |
| `/api/compress` | POST | Compress uploaded file |
| `/api/decompress` | POST | Decompress file |
| `/api/benchmark` | POST | Run compression benchmarks |
| `/api/upload` | POST | Upload neural data file |
| `/api/download/{filename}` | GET | Download processed file |
| `/api/generate-data` | POST | Generate synthetic test data |

### WebSocket API

**Endpoint:** `/ws/metrics`

**Commands:**
```json
{
  "command": "start_stream",
  "plugin": "huffman",
  "mode": "balanced",
  "quality": 0.8
}

{
  "command": "stop_stream",
  "session_id": "session_uuid"
}
```

**Responses:**
```json
{
  "type": "metrics",
  "data": {
    "timestamp": 1645123456.789,
    "session_id": "uuid",
    "compression_ratio": 3.2,
    "latency_ms": 0.5,
    "snr_db": 45.2,
    "psnr_db": 52.1,
    "gpu_available": true
  }
}
```

## Usage Guide

### Live Stream Monitoring

1. **Select Plugin**: Choose compression algorithm from dropdown
2. **Configure Settings**: Set mode (fast/balanced/quality) and quality level
3. **Start Stream**: Click "Start Stream" to begin real-time monitoring
4. **Monitor Metrics**: View live charts and real-time performance indicators
5. **Stop Stream**: Click "Stop Stream" when done

### File Compression

1. **Select File**: Upload .npy or .h5 neural data file
2. **Choose Plugin**: Select compression algorithm and settings
3. **Compress**: Click "Compress File" to process
4. **Review Results**: View compression metrics and quality scores
5. **Download**: Get compressed file or decompress and download original

### Benchmarking

1. **Select Plugins**: Choose multiple algorithms to compare
2. **Configure Tests**: Set compression mode and test parameters
3. **Generate Data**: Create synthetic test data if needed
4. **Run Benchmark**: Execute automated comparison tests
5. **Analyze Results**: Review performance charts and trade-off analysis

## Architecture

### Backend (FastAPI)
- **Plugin Discovery**: Automatic detection of bci_compression plugins
- **WebSocket Manager**: Real-time metrics broadcasting
- **File Handling**: Upload/download with progress tracking
- **Synthetic Data**: Neural signal generation for testing

### Frontend (React + TypeScript)
- **Component Architecture**: Modular design with shared components
- **Real-Time Charts**: Recharts integration for live visualization
- **State Management**: React hooks and context for data flow
- **Responsive Design**: Mobile-first CSS framework

### Data Flow
1. User selects plugin and settings in UI
2. Frontend sends configuration via WebSocket
3. Backend processes neural data with selected algorithm
4. Metrics streamed back via WebSocket
5. Charts update in real-time with performance data

## Development

### Project Structure
```
dashboard/
├── src/
│   ├── components/          # React components
│   │   ├── LiveStream.tsx   # Real-time monitoring
│   │   ├── FileCompression.tsx  # File operations
│   │   ├── Benchmarks.tsx   # Performance testing
│   │   └── PluginPicker.tsx # Shared plugin selector
│   ├── contexts/            # React context providers
│   ├── types/               # TypeScript interfaces
│   └── tests/               # Frontend tests
scripts/
└── telemetry_server.py      # FastAPI backend
tests/
└── test_telemetry_server.py # Backend tests
```

### Testing

**Frontend Tests:**
```bash
cd dashboard
npm test
```

**Backend Tests:**
```bash
pytest tests/test_telemetry_server.py
```

### Configuration

**Environment Variables:**
```bash
# Backend configuration
BCI_COMPRESSION_PLUGINS_PATH=/path/to/plugins
UPLOAD_DIR=/tmp/bci_uploads
MAX_FILE_SIZE_MB=100

# Frontend configuration (optional)
VITE_API_BASE_URL=http://localhost:8000
```

## Performance Considerations

### Real-Time Requirements
- Target < 1ms latency for streaming applications
- Efficient buffering for continuous data streams
- GPU acceleration when available
- Memory optimization for large files

### Scalability
- Concurrent WebSocket connections supported
- File upload size limits configurable
- Automatic cleanup of temporary files
- Connection pooling for database operations

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

### Common Issues

**WebSocket Connection Failed**
- Check backend server is running on port 8000
- Verify firewall settings allow WebSocket connections
- Ensure browser supports WebSocket API

**File Upload Errors**
- Check file format (.npy or .h5 only)
- Verify file size under configured limit
- Ensure sufficient disk space for uploads

**Plugin Not Found**
- Verify bci_compression package is installed
- Check plugin path configuration
- Ensure plugin implements required interface

### Debug Mode

Enable debug logging:
```bash
# Backend
PYTHONPATH=. uvicorn scripts.telemetry_server:app --log-level debug

# Frontend
npm run dev -- --debug
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Run tests: `npm test && pytest`
4. Submit pull request with description

## License

MIT License - see LICENSE file for details.
