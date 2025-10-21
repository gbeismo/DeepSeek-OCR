# DeepSeek OCR Web Interface Guide

A beautiful, easy-to-use web interface for batch processing images with DeepSeek OCR. Simply drag and drop your images and get OCR results instantly!

## Features

- 🖱️ **Drag & Drop Interface**: Intuitive drag-and-drop file upload
- 📊 **Real-time Progress**: Live progress tracking for batch processing
- 🎨 **Beautiful UI**: Modern, responsive design that works on all devices
- ⚙️ **Flexible Settings**: Choose backend, mode, and custom prompts
- 📥 **Easy Downloads**: Download results as markdown files
- 📋 **Copy to Clipboard**: Quick copy functionality for OCR results
- 🔄 **Batch Processing**: Process multiple images at once

## Quick Start

### 1. Install Dependencies

```bash
# Install web server dependencies
pip install -r requirements_web.txt

# Make sure you have the main requirements installed too
pip install -r requirements.txt
```

### 2. Start the Web Server

```bash
python web_server.py
```

By default, the server runs on `http://localhost:5000`

### 3. Open the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

### 4. Upload and Process Images

1. **Drag and drop** images onto the upload zone (or click to browse)
2. **Select settings**:
   - Choose backend (vLLM or Transformers)
   - Choose processing mode (Tiny, Small, Base, Large, or Gundam)
   - Select or customize the prompt
3. **Click "Process"** and wait for results
4. **Download or copy** the OCR results

## Server Configuration

### Custom Port and Host

```bash
# Run on a different port
python web_server.py --port 8080

# Allow external connections
python web_server.py --host 0.0.0.0 --port 8080
```

### Set Default Backend and Mode

```bash
# Use Transformers backend by default
python web_server.py --backend transformers

# Use Base mode by default
python web_server.py --mode base
```

### Debug Mode

```bash
# Enable debug mode for development
python web_server.py --debug
```

### All Options

```bash
python web_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --backend vllm \
  --mode gundam \
  --debug
```

## Using the Interface

### Upload Methods

1. **Drag & Drop**: Drag image files from your file explorer directly into the upload zone
2. **Click to Browse**: Click on the upload zone to open a file picker
3. **Multiple Files**: Select or drop multiple files at once for batch processing

### Supported Formats

- JPG/JPEG
- PNG
- GIF
- BMP
- TIFF
- WebP

Maximum file size: 50MB per file

### Processing Settings

#### Backend Options

- **vLLM** (Recommended): Fast, GPU-optimized backend for batch processing
  - Best for: Processing multiple images quickly
  - Requires: vLLM installation

- **Transformers**: Simple backend using HuggingFace Transformers
  - Best for: Single images or simpler setup
  - Requires: Transformers library

#### Processing Modes

| Mode | Resolution | Speed | Quality | Best For |
|------|-----------|-------|---------|----------|
| **Gundam** | Dynamic | Adaptive | High | Complex documents (Recommended) |
| **Large** | 1280×1280 | Slow | Highest | High-quality scans |
| **Base** | 1024×1024 | Medium | Good | Standard documents |
| **Small** | 640×640 | Fast | Fair | Simple documents |
| **Tiny** | 512×512 | Fastest | Basic | Quick previews |

### Preset Prompts

Click the preset buttons to quickly select common prompts:

- **📄 Markdown**: Convert document to markdown (preserves layout)
- **📝 Free OCR**: Simple text extraction without layout
- **📊 Parse Figure**: Extract data from charts and figures
- **🖼️ Describe Image**: Get detailed image descriptions

Or write your own custom prompt!

### Viewing Results

Once processing is complete:

1. **Results appear automatically** below the upload section
2. **Preview**: See a preview of the OCR output
3. **Download**: Download the full markdown file
4. **Copy**: Copy the text directly to clipboard

## API Endpoints

The web server also provides a REST API for programmatic access:

### Upload Files

```bash
POST /api/upload
Content-Type: multipart/form-data

files[]: image files
prompt: OCR prompt
backend: vllm or transformers
mode: tiny, small, base, large, or gundam
```

### Check Job Status

```bash
GET /api/job/{job_id}

Returns:
{
  "id": "job-uuid",
  "status": "queued|processing|completed|failed",
  "file_count": 5,
  "created_at": "2025-10-21T...",
  ...
}
```

### Get Results

```bash
GET /api/job/{job_id}/results

Returns:
{
  "results": [
    {
      "image_path": "...",
      "content": "OCR content...",
      "clean_output_path": "..."
    }
  ]
}
```

### Download Result File

```bash
GET /api/job/{job_id}/download/{filename}
```

### Get Available Modes

```bash
GET /api/modes

Returns list of processing modes with descriptions
```

### Get Preset Prompts

```bash
GET /api/prompts

Returns list of preset prompts
```

### Health Check

```bash
GET /api/health

Returns:
{
  "status": "healthy",
  "timestamp": "..."
}
```

## Using the API Programmatically

### Python Example

```python
import requests

# Upload files
files = [
    ('files[]', open('image1.jpg', 'rb')),
    ('files[]', open('image2.jpg', 'rb'))
]

data = {
    'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
    'backend': 'vllm',
    'mode': 'gundam'
}

response = requests.post('http://localhost:5000/api/upload', files=files, data=data)
job_id = response.json()['job_id']

# Check status
while True:
    status_response = requests.get(f'http://localhost:5000/api/job/{job_id}')
    status = status_response.json()['status']

    if status == 'completed':
        break
    elif status == 'failed':
        print("Processing failed")
        break

    time.sleep(2)

# Get results
results = requests.get(f'http://localhost:5000/api/job/{job_id}/results')
print(results.json())
```

### cURL Example

```bash
# Upload files
curl -X POST http://localhost:5000/api/upload \
  -F "files[]=@image1.jpg" \
  -F "files[]=@image2.jpg" \
  -F "prompt=<image>\n<|grounding|>Convert the document to markdown." \
  -F "backend=vllm" \
  -F "mode=gundam"

# Returns: {"job_id": "...", "status": "queued", "file_count": 2}

# Check status
curl http://localhost:5000/api/job/{job_id}

# Get results
curl http://localhost:5000/api/job/{job_id}/results

# Download a result file
curl -O http://localhost:5000/api/job/{job_id}/download/result.md
```

### JavaScript/Fetch Example

```javascript
// Upload files
const formData = new FormData();
formData.append('files[]', file1);
formData.append('files[]', file2);
formData.append('prompt', '<image>\n<|grounding|>Convert the document to markdown.');
formData.append('backend', 'vllm');
formData.append('mode', 'gundam');

const response = await fetch('http://localhost:5000/api/upload', {
    method: 'POST',
    body: formData
});

const { job_id } = await response.json();

// Poll for results
const pollStatus = async () => {
    const response = await fetch(`http://localhost:5000/api/job/${job_id}`);
    const data = await response.json();

    if (data.status === 'completed') {
        const results = await fetch(`http://localhost:5000/api/job/${job_id}/results`);
        return await results.json();
    } else if (data.status === 'failed') {
        throw new Error('Processing failed');
    } else {
        await new Promise(resolve => setTimeout(resolve, 2000));
        return pollStatus();
    }
};

const results = await pollStatus();
```

## Directory Structure

When the web server runs, it creates the following directories:

```
DeepSeek-OCR/
├── uploads/              # Uploaded files (organized by job ID)
│   └── {job-id}/
│       ├── image1.jpg
│       └── image2.jpg
├── outputs/              # OCR results (organized by job ID)
│   └── {job-id}/
│       ├── image1.md
│       ├── image1_raw.md
│       ├── image2.md
│       ├── image2_raw.md
│       └── batch_metadata.json
└── web_server.py
```

## Troubleshooting

### Server Won't Start

**Issue**: `ModuleNotFoundError: No module named 'flask'`

**Solution**: Install web dependencies:
```bash
pip install -r requirements_web.txt
```

### Can't Connect to Server

**Issue**: Browser shows "Can't connect" or "Connection refused"

**Solutions**:
1. Make sure the server is running: `python web_server.py`
2. Check the correct port: `http://localhost:5000`
3. Check firewall settings if accessing remotely

### Upload Fails

**Issue**: Files upload but processing fails

**Solutions**:
1. Check that the OCR model is installed correctly
2. Verify GPU is available (for vLLM backend)
3. Try switching to Transformers backend
4. Check server logs for error messages

### Out of Memory

**Issue**: CUDA out of memory errors

**Solutions**:
1. Use smaller processing mode (Tiny or Small)
2. Process fewer files at once
3. Switch to Transformers backend
4. Reduce `max_concurrency` in code

### Processing is Slow

**Solutions**:
1. Use vLLM backend instead of Transformers for batch processing
2. Use Gundam or Small mode for faster processing
3. Ensure GPU is being used (check with `nvidia-smi`)
4. Process smaller batches

## Security Considerations

### Production Deployment

If deploying in production:

1. **Add Authentication**: The current server has no authentication
2. **Use HTTPS**: Set up SSL/TLS certificates
3. **Limit File Sizes**: Adjust `MAX_CONTENT_LENGTH` as needed
4. **Set CORS Properly**: Configure `flask-cors` for your domain
5. **Use a Production Server**: Use gunicorn or uWSGI instead of Flask dev server

### Example with Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_server:app
```

### Environment Variables

```bash
# Set upload limits
export MAX_CONTENT_LENGTH=104857600  # 100MB

# Set custom upload/output directories
export UPLOAD_FOLDER=/path/to/uploads
export OUTPUT_FOLDER=/path/to/outputs
```

## Tips for Best Results

1. **Use High-Quality Images**: Better image quality = better OCR results
2. **Choose the Right Mode**: Gundam mode adapts to your image automatically
3. **Customize Prompts**: Tailor prompts to your specific use case
4. **Batch Similar Images**: Process similar documents together with the same settings
5. **Check Previews**: Review the preview before downloading full results

## Advanced Usage

### Custom Prompts

Create custom prompts for specialized tasks:

```
# Extract specific information
<image>
<|grounding|>Extract all dates, names, and amounts from this invoice.

# Translate while OCR-ing
<image>
<|grounding|>Convert this document to markdown and translate to English.

# Focus on specific regions
<image>
Locate <|ref|>table of contents<|/ref|> in the image.
```

### Processing Large Batches

For very large batches (100+ images):

1. Use vLLM backend
2. Use Tiny or Small mode for speed
3. Process in chunks of 50-100 images
4. Monitor GPU memory usage

## Support

For issues or questions:

1. Check the main [BATCH_OCR_GUIDE.md](BATCH_OCR_GUIDE.md)
2. Review the [README.md](README.md)
3. Check server logs for error messages
4. Verify all dependencies are installed correctly

## License

Same as DeepSeek-OCR main repository.
