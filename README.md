<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek AI" />
</div>


<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="assets/badge.svg" />
  </a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>

</div>



<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR"><b>📥 Model Download</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>📄 Paper Link</b></a> |
  <a href="./DeepSeek_OCR_paper.pdf"><b>📄 Arxiv Paper Link</b></a> |
</p>

<h2>
<p align="center">
  <a href="">DeepSeek-OCR: Contexts Optical Compression</a>
</p>
</h2>

<p align="center">
<img src="assets/fig1.png" style="width: 1000px" align=center>
</p>
<p align="center">
<a href="">Explore the boundaries of visual-text compression.</a>       
</p>

## Release
- [2025/10/20]🚀🚀🚀 We release DeepSeek-OCR, a model to investigate the role of vision encoders from an LLM-centric viewpoint.

## Contents
- [Install](#install)
- [Web Interface](#web-interface) (NEW! - Easiest way to use)
- [Batch Processing Wrapper](#batch-processing-wrapper) (NEW!)
- [vLLM Inference](#vllm-inference)
- [Transformers Inference](#transformers-inference)
  




## Install
>Our environment is cuda11.8+torch2.6.0.
1. Clone this repository and navigate to the DeepSeek-OCR folder
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
```
2. Conda
```Shell
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```
3. Packages

- download the vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) 
```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```
**Note:** if you want vLLM and transformers codes to run in the same environment, you don't need to worry about this installation error like: vllm 0.8.5+cu118 requires transformers>=4.51.1

## Web Interface

We provide a beautiful, easy-to-use web interface for batch processing images. Simply drag and drop your images in a browser!

### Quick Start

1. **Install web dependencies**:
```bash
pip install -r requirements_web.txt
```

2. **Start the web server**:
```bash
python web_server.py
```

3. **Open in browser**:
```
http://localhost:5000
```

4. **Drag & drop images** and click "Process"!

### Features

- 🖱️ **Drag & Drop Interface**: Intuitive file upload
- 📊 **Real-time Progress**: Live progress tracking
- 🎨 **Beautiful UI**: Modern, responsive design
- ⚙️ **Flexible Settings**: Choose backend, mode, and prompts
- 📥 **Easy Downloads**: Download markdown results
- 📋 **Copy to Clipboard**: Quick copy functionality

### Screenshots

The web interface provides:
- Drag-and-drop image upload
- Processing mode selection (Tiny, Small, Base, Large, Gundam)
- Backend selection (vLLM or Transformers)
- Preset prompts (Markdown, Free OCR, Parse Figure, etc.)
- Real-time progress tracking
- Instant result preview and download

### Server Options

```bash
# Run on custom port
python web_server.py --port 8080

# Allow external connections
python web_server.py --host 0.0.0.0

# Set default backend
python web_server.py --backend transformers --mode base
```

### REST API

The web server also provides a REST API:

```python
import requests

# Upload files
files = [('files[]', open('image.jpg', 'rb'))]
data = {'prompt': '<image>\n<|grounding|>Convert to markdown.', 'backend': 'vllm'}
response = requests.post('http://localhost:5000/api/upload', files=files, data=data)
job_id = response.json()['job_id']

# Get results
results = requests.get(f'http://localhost:5000/api/job/{job_id}/results')
```

**Full documentation**: [WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md)

## Batch Processing Wrapper

We provide a user-friendly wrapper for batch processing multiple images with DeepSeek OCR. This tool makes it easy to process entire directories of images with a simple command or Python API.

### Quick Start

#### Command Line Interface

Process all images in a directory:
```bash
# Using vLLM backend (recommended for batch processing)
python batch_ocr.py --input-dir ./images --output-dir ./results

# Using Transformers backend (simpler setup)
python batch_ocr.py --input-dir ./images --output-dir ./results --backend transformers

# Custom processing mode
python batch_ocr.py --input-dir ./images --output-dir ./results --mode gundam

# Custom prompt
python batch_ocr.py --input-dir ./images --output-dir ./results --prompt "<image>\nFree OCR."
```

#### Python API

```python
from batch_ocr import DeepSeekOCRWrapper

# Create wrapper
wrapper = DeepSeekOCRWrapper(backend="vllm", mode="gundam")

# Process directory
results = wrapper.process_directory(
    input_dir="./images",
    output_dir="./results",
    prompt="<image>\n<|grounding|>Convert the document to markdown."
)

print(f"Processed {len(results)} images")
```

### Features

- **Dual Backend Support**: Choose between vLLM (fast) or Transformers (simple)
- **Batch Processing**: Process multiple images in parallel with progress tracking
- **Multiple Modes**: Tiny, Small, Base, Large, and Gundam (dynamic resolution)
- **Flexible Prompts**: Customize for documents, figures, or general images
- **Auto Cleanup**: Removes detection tags and formats output
- **Progress Tracking**: Real-time progress bars with tqdm

### Processing Modes

| Mode | Resolution | Vision Tokens | Use Case |
|------|-----------|---------------|----------|
| Tiny | 512×512 | 64 | Quick preview, simple text |
| Small | 640×640 | 100 | Simple documents |
| Base | 1024×1024 | 256 | Standard documents |
| Large | 1280×1280 | 400 | High-quality documents |
| Gundam | Dynamic | Variable | Complex layouts (Recommended) |

### Examples

See the `examples/` directory for complete examples:
- `example_basic_batch.py` - Simple batch processing
- `example_multiple_modes.py` - Compare different modes
- `example_custom_prompts.py` - Different prompts for different tasks
- `example_transformers_backend.py` - Using Transformers backend

### Documentation

For complete documentation, see:
- [BATCH_OCR_GUIDE.md](BATCH_OCR_GUIDE.md) - Complete batch processing guide
- [examples/README.md](examples/README.md) - Example usage guide

## vLLM-Inference
- VLLM:
>**Note:** change the INPUT_PATH/OUTPUT_PATH and other settings in the DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
```
1. image: streaming output
```Shell
python run_dpsk_ocr_image.py
```
2. pdf: concurrency ~2500tokens/s(an A100-40G)
```Shell
python run_dpsk_ocr_pdf.py
```
3. batch eval for benchmarks
```Shell
python run_dpsk_ocr_eval_batch.py
```
## Transformers-Inference
- Transformers
```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
```
or you can
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```
## Support-Modes
The current open-source model supports the following modes:
- Native resolution:
  - Tiny: 512×512 （64 vision tokens）✅
  - Small: 640×640 （100 vision tokens）✅
  - Base: 1024×1024 （256 vision tokens）✅
  - Large: 1280×1280 （400 vision tokens）✅
- Dynamic resolution
  - Gundam: n×640×640 + 1×1024×1024 ✅

## Prompts examples
```python
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
```


## Visualizations
<table>
<tr>
<td><img src="assets/show1.jpg" style="width: 500px"></td>
<td><img src="assets/show2.jpg" style="width: 500px"></td>
</tr>
<tr>
<td><img src="assets/show3.jpg" style="width: 500px"></td>
<td><img src="assets/show4.jpg" style="width: 500px"></td>
</tr>
</table>


## Acknowledgement

We would like to thank [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OneChart](https://github.com/LingyvKong/OneChart), [Slow Perception](https://github.com/Ucas-HaoranWei/Slow-Perception) for their valuable models and ideas.

We also appreciate the benchmarks: [Fox](https://github.com/ucaslcl/Fox), [OminiDocBench](https://github.com/opendatalab/OmniDocBench).

## Citation

coming soon！







