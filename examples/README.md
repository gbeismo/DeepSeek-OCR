# DeepSeek OCR Examples

This directory contains example scripts demonstrating different ways to use the DeepSeek OCR batch processing wrapper.

## Setup

Before running examples, create an `input_images` directory with some test images:

```bash
mkdir input_images
cp /path/to/your/test/images/*.jpg input_images/
```

## Available Examples

### 1. Basic Batch Processing (`example_basic_batch.py`)

The simplest example - process a directory of images with default settings.

```bash
python example_basic_batch.py
```

**What it demonstrates:**
- Creating an OCR wrapper
- Processing a directory of images
- Accessing results

### 2. Multiple Processing Modes (`example_multiple_modes.py`)

Compare different processing modes (tiny, small, base, gundam) to see speed/quality tradeoffs.

```bash
python example_multiple_modes.py
```

**What it demonstrates:**
- Using different processing modes
- Timing comparisons
- Quality vs speed tradeoffs

### 3. Custom Prompts (`example_custom_prompts.py`)

Process images with different prompts for different use cases.

```bash
python example_custom_prompts.py
```

**What it demonstrates:**
- Markdown conversion
- Free OCR
- Figure parsing
- Image description
- Different prompt templates

### 4. Transformers Backend (`example_transformers_backend.py`)

Use the Transformers backend instead of vLLM.

```bash
python example_transformers_backend.py
```

**What it demonstrates:**
- Using Transformers backend
- Simpler setup (no vLLM required)
- Good for single image processing

## Running All Examples

```bash
# Create test directory
mkdir -p input_images

# Add some test images
# (copy your own images to input_images/)

# Run examples
python example_basic_batch.py
python example_multiple_modes.py
python example_custom_prompts.py
python example_transformers_backend.py
```

## Expected Output Structure

After running the examples, you'll have:

```
examples/
├── input_images/              # Your test images
├── ocr_results/              # Basic batch results
├── mode_comparison/          # Results from different modes
│   ├── tiny/
│   ├── small/
│   ├── base/
│   └── gundam/
├── prompt_examples/          # Results from different prompts
│   ├── markdown_conversion/
│   ├── free_ocr/
│   ├── parse_figure/
│   ├── detailed_description/
│   └── ocr_image/
└── transformers_output/      # Results from Transformers backend
```

## Modifying Examples

All examples are well-commented and easy to modify. Common modifications:

### Change GPU Device

```python
wrapper = DeepSeekOCRWrapper(
    gpu_id="1"  # Use GPU 1 instead of 0
)
```

### Change Processing Mode

```python
wrapper = DeepSeekOCRWrapper(
    mode="large"  # Use large mode for higher quality
)
```

### Use Custom Model Path

```python
wrapper = DeepSeekOCRWrapper(
    model_path="/path/to/your/model"
)
```

### Adjust Concurrency

```python
wrapper = DeepSeekOCRWrapper(
    max_concurrency=50,  # Lower if you have limited GPU memory
    num_workers=32
)
```

## Troubleshooting

### Issue: "No images found"

Make sure you've created the `input_images` directory and added some images:

```bash
mkdir input_images
cp /path/to/images/*.jpg input_images/
```

### Issue: CUDA out of memory

Try a smaller mode or lower concurrency:

```python
wrapper = DeepSeekOCRWrapper(
    mode="small",
    max_concurrency=10
)
```

### Issue: vLLM not installed

Use the Transformers backend:

```python
wrapper = DeepSeekOCRWrapper(
    backend="transformers"
)
```

## Next Steps

After exploring these examples:

1. Try processing your own documents
2. Experiment with different prompts
3. Compare different modes for your use case
4. Integrate the wrapper into your own projects

See `../BATCH_OCR_GUIDE.md` for complete documentation.
