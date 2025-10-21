# DeepSeek OCR Batch Processing Guide

A comprehensive wrapper for batch processing images with DeepSeek OCR. This tool provides a simple CLI and Python API for processing multiple images efficiently.

## Features

- **Dual Backend Support**: Choose between vLLM (fast, GPU-optimized) or Transformers (simple setup)
- **Batch Processing**: Process multiple images in parallel with progress tracking
- **Multiple Processing Modes**: Tiny, Small, Base, Large, and Gundam (dynamic resolution)
- **Flexible Prompts**: Customize OCR behavior for different use cases
- **Automatic Cleanup**: Removes detection tags and formats output
- **Progress Tracking**: Real-time progress bars with tqdm
- **Metadata Export**: JSON metadata for batch processing results

## Quick Start

### Command Line Usage

#### 1. Basic Usage (vLLM Backend - Recommended)

```bash
python batch_ocr.py --input-dir ./my_images --output-dir ./ocr_results
```

This will:
- Process all images in `./my_images`
- Use vLLM backend (fastest)
- Use Gundam mode (dynamic resolution)
- Save results to `./ocr_results`

#### 2. Transformers Backend (Simpler Setup)

```bash
python batch_ocr.py \
  --input-dir ./my_images \
  --output-dir ./ocr_results \
  --backend transformers
```

Use this if you don't have vLLM installed or want a simpler setup.

#### 3. Different Processing Modes

```bash
# Tiny mode (fast, lower quality)
python batch_ocr.py -i ./images -o ./results --mode tiny

# Small mode
python batch_ocr.py -i ./images -o ./results --mode small

# Base mode (balanced)
python batch_ocr.py -i ./images -o ./results --mode base

# Large mode (slow, higher quality)
python batch_ocr.py -i ./images -o ./results --mode large

# Gundam mode (dynamic resolution - recommended)
python batch_ocr.py -i ./images -o ./results --mode gundam
```

#### 4. Custom Prompts

```bash
# Free OCR (no layout preservation)
python batch_ocr.py -i ./images -o ./results \
  --prompt "<image>\nFree OCR."

# Parse figures
python batch_ocr.py -i ./images -o ./results \
  --prompt "<image>\nParse the figure."

# Detailed description
python batch_ocr.py -i ./images -o ./results \
  --prompt "<image>\nDescribe this image in detail."
```

#### 5. Advanced Options

```bash
python batch_ocr.py \
  --input-dir ./images \
  --output-dir ./results \
  --backend vllm \
  --mode gundam \
  --extensions jpg png pdf \
  --gpu-id 0 \
  --max-concurrency 50 \
  --num-workers 32
```

### List Available Modes

```bash
python batch_ocr.py --list-modes
```

Output:
```
Available Processing Modes:
------------------------------------------------------------
tiny     - Tiny mode: 512x512, 64 vision tokens
small    - Small mode: 640x640, 100 vision tokens
base     - Base mode: 1024x1024, 256 vision tokens
large    - Large mode: 1280x1280, 400 vision tokens
gundam   - Gundam mode: Dynamic resolution (n×640×640 + 1×1024×1024)
```

## Python API Usage

### Example 1: Simple Batch Processing

```python
from batch_ocr import DeepSeekOCRWrapper

# Create wrapper
wrapper = DeepSeekOCRWrapper(
    backend="vllm",
    mode="gundam",
    gpu_id="0"
)

# Process directory
results = wrapper.process_directory(
    input_dir="./images",
    output_dir="./results",
    prompt="<image>\n<|grounding|>Convert the document to markdown."
)

print(f"Processed {len(results)} images")
```

### Example 2: Custom Configuration

```python
from batch_ocr import DeepSeekOCRWrapper

# Advanced configuration
wrapper = DeepSeekOCRWrapper(
    model_path="deepseek-ai/DeepSeek-OCR",
    backend="vllm",
    mode="base",
    gpu_id="0",
    max_concurrency=100,
    num_workers=64
)

# Process with custom settings
results = wrapper.process_directory(
    input_dir="./documents",
    output_dir="./ocr_output",
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    extensions=["jpg", "png", "tiff"],
    save_metadata=True
)

# Access results
for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Output: {result['clean_output_path']}")
    print(f"Content preview: {result['content'][:100]}...")
```

### Example 3: Different Backends

```python
from batch_ocr import DeepSeekOCRWrapper

# vLLM backend (fastest for batch processing)
vllm_wrapper = DeepSeekOCRWrapper(backend="vllm", mode="gundam")
vllm_results = vllm_wrapper.process_directory("./images", "./vllm_output")

# Transformers backend (simpler, good for single images)
transformers_wrapper = DeepSeekOCRWrapper(backend="transformers", mode="base")
transformers_results = transformers_wrapper.process_directory("./images", "./transformers_output")
```

### Example 4: Processing Modes

```python
from batch_ocr import DeepSeekOCRWrapper, OCRModes

# List all available modes
for mode in OCRModes.list_modes():
    print(f"{mode.name}: {mode.description}")

# Use specific mode
wrapper = DeepSeekOCRWrapper(mode="large")
results = wrapper.process_directory("./high_res_docs", "./results")
```

## Processing Modes

| Mode | Resolution | Vision Tokens | Use Case | Speed |
|------|-----------|---------------|----------|-------|
| **Tiny** | 512×512 | 64 | Quick preview, simple text | Fastest |
| **Small** | 640×640 | 100 | Simple documents | Fast |
| **Base** | 1024×1024 | 256 | Standard documents | Balanced |
| **Large** | 1280×1280 | 400 | High-quality documents | Slow |
| **Gundam** | Dynamic | Variable | Complex layouts (Recommended) | Adaptive |

## Prompt Templates

### Document Processing

```bash
# Convert document to markdown (preserves layout)
--prompt "<image>\n<|grounding|>Convert the document to markdown."

# Free OCR (text only, no layout)
--prompt "<image>\nFree OCR."
```

### Image Analysis

```bash
# Parse figures and charts
--prompt "<image>\nParse the figure."

# Detailed image description
--prompt "<image>\nDescribe this image in detail."

# Locate specific text
--prompt "<image>\nLocate <|ref|>your text here<|/ref|> in the image."
```

### Special Cases

```bash
# OCR general images (not documents)
--prompt "<image>\n<|grounding|>OCR this image."
```

## Output Structure

When you run batch processing, the output directory will contain:

```
output_dir/
├── batch_metadata.json          # Metadata about the batch processing
├── image1.md                    # Cleaned OCR result
├── image1_raw.md                # Raw OCR output with detection tags
├── image2.md
├── image2_raw.md
└── ...
```

### Metadata Format

The `batch_metadata.json` contains:

```json
{
  "total_images": 10,
  "mode": "gundam",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "backend": "vllm",
  "results": [
    {
      "image_path": "/path/to/image1.jpg",
      "raw_output_path": "/path/to/output/image1_raw.md",
      "clean_output_path": "/path/to/output/image1.md",
      "content": "OCR content here..."
    }
  ]
}
```

## Performance Tips

### For Maximum Speed (vLLM)

```bash
python batch_ocr.py \
  -i ./images \
  -o ./results \
  --backend vllm \
  --mode gundam \
  --max-concurrency 100 \
  --num-workers 64
```

### For Limited GPU Memory

```bash
python batch_ocr.py \
  -i ./images \
  -o ./results \
  --backend vllm \
  --mode small \
  --max-concurrency 20 \
  --num-workers 16
```

### For CPU/Simple Setup

```bash
python batch_ocr.py \
  -i ./images \
  -o ./results \
  --backend transformers \
  --mode tiny
```

## Common Use Cases

### 1. Process Research Papers

```bash
python batch_ocr.py \
  -i ./papers \
  -o ./papers_ocr \
  --mode gundam \
  --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

### 2. Extract Text from Screenshots

```bash
python batch_ocr.py \
  -i ./screenshots \
  -o ./screenshots_text \
  --mode small \
  --prompt "<image>\nFree OCR."
```

### 3. Parse Charts and Figures

```bash
python batch_ocr.py \
  -i ./figures \
  -o ./figures_parsed \
  --mode base \
  --prompt "<image>\nParse the figure."
```

### 4. High-Quality Document Digitization

```bash
python batch_ocr.py \
  -i ./scanned_docs \
  -o ./digitized \
  --mode large \
  --backend vllm \
  --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

## Troubleshooting

### Issue: "vLLM not installed"

**Solution**: Install vLLM or use transformers backend:
```bash
# Option 1: Install vLLM (see main README.md)
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Option 2: Use transformers backend
python batch_ocr.py ... --backend transformers
```

### Issue: CUDA out of memory

**Solution**: Reduce concurrency or use smaller mode:
```bash
python batch_ocr.py \
  ... \
  --mode small \
  --max-concurrency 10 \
  --num-workers 8
```

### Issue: No images found

**Solution**: Check extensions or add more:
```bash
python batch_ocr.py \
  ... \
  --extensions jpg jpeg png bmp tiff
```

## Requirements

- Python 3.8+
- PyTorch 2.6.0+
- One of:
  - vLLM 0.8.5 (for vLLM backend)
  - Transformers 4.51.1+ (for Transformers backend)
- See main README.md for full installation instructions

## License

Same as DeepSeek-OCR main repository.
