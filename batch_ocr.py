#!/usr/bin/env python3
"""
DeepSeek OCR Batch Processing Wrapper

A user-friendly wrapper for batch processing images with DeepSeek OCR.
Supports both vLLM (fast) and Transformers (simple) backends.

Usage:
    # Process all images in a directory (vLLM backend - fastest)
    python batch_ocr.py --input-dir ./images --output-dir ./results --backend vllm

    # Process with transformers backend (simpler, no vLLM required)
    python batch_ocr.py --input-dir ./images --output-dir ./results --backend transformers

    # Use specific processing mode
    python batch_ocr.py --input-dir ./images --output-dir ./results --mode gundam

    # Custom prompt
    python batch_ocr.py --input-dir ./images --output-dir ./results --prompt "<image>\nFree OCR."

    # Process specific file types
    python batch_ocr.py --input-dir ./images --output-dir ./results --extensions jpg png
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from tqdm import tqdm


@dataclass
class ProcessingMode:
    """OCR processing mode configuration"""
    name: str
    base_size: int
    image_size: int
    crop_mode: bool
    description: str


class OCRModes:
    """Available OCR processing modes"""

    TINY = ProcessingMode(
        name="tiny",
        base_size=512,
        image_size=512,
        crop_mode=False,
        description="Tiny mode: 512x512, 64 vision tokens"
    )

    SMALL = ProcessingMode(
        name="small",
        base_size=640,
        image_size=640,
        crop_mode=False,
        description="Small mode: 640x640, 100 vision tokens"
    )

    BASE = ProcessingMode(
        name="base",
        base_size=1024,
        image_size=1024,
        crop_mode=False,
        description="Base mode: 1024x1024, 256 vision tokens"
    )

    LARGE = ProcessingMode(
        name="large",
        base_size=1280,
        image_size=1280,
        crop_mode=False,
        description="Large mode: 1280x1280, 400 vision tokens"
    )

    GUNDAM = ProcessingMode(
        name="gundam",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        description="Gundam mode: Dynamic resolution (n×640×640 + 1×1024×1024)"
    )

    @classmethod
    def get_mode(cls, mode_name: str) -> ProcessingMode:
        """Get processing mode by name"""
        modes = {
            "tiny": cls.TINY,
            "small": cls.SMALL,
            "base": cls.BASE,
            "large": cls.LARGE,
            "gundam": cls.GUNDAM,
        }
        return modes.get(mode_name.lower(), cls.GUNDAM)

    @classmethod
    def list_modes(cls) -> List[ProcessingMode]:
        """List all available modes"""
        return [cls.TINY, cls.SMALL, cls.BASE, cls.LARGE, cls.GUNDAM]


class DeepSeekOCRWrapper:
    """Wrapper for batch processing with DeepSeek OCR"""

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        backend: str = "vllm",
        mode: str = "gundam",
        gpu_id: str = "0",
        max_concurrency: int = 100,
        num_workers: int = 64,
    ):
        """
        Initialize OCR wrapper

        Args:
            model_path: Path to DeepSeek OCR model
            backend: Processing backend ('vllm' or 'transformers')
            mode: Processing mode (tiny, small, base, large, gundam)
            gpu_id: GPU device ID
            max_concurrency: Maximum concurrent requests (vLLM only)
            num_workers: Number of worker threads for preprocessing
        """
        self.model_path = model_path
        self.backend = backend.lower()
        self.mode_config = OCRModes.get_mode(mode)
        self.gpu_id = gpu_id
        self.max_concurrency = max_concurrency
        self.num_workers = num_workers

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        if backend == "vllm":
            os.environ['VLLM_USE_V1'] = '0'
            import torch
            if torch.version.cuda == '11.8':
                os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

        self.model = None
        self.tokenizer = None

    def _init_vllm_backend(self):
        """Initialize vLLM backend"""
        try:
            from vllm import LLM, SamplingParams
            from vllm.model_executor.models.registry import ModelRegistry
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))
            from deepseek_ocr import DeepseekOCRForCausalLM
            from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
            from process.image_process import DeepseekOCRProcessor

            ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

            print("Initializing vLLM model...")
            self.llm = LLM(
                model=self.model_path,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                enforce_eager=False,
                trust_remote_code=True,
                max_model_len=8192,
                swap_space=0,
                max_num_seqs=self.max_concurrency,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            )

            logits_processors = [
                NoRepeatNGramLogitsProcessor(
                    ngram_size=40,
                    window_size=90,
                    whitelist_token_ids={128821, 128822}
                )
            ]

            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
            )

            self.processor = DeepseekOCRProcessor()
            print("vLLM backend initialized successfully")

        except ImportError as e:
            print(f"Error: vLLM backend requires vLLM to be installed: {e}")
            print("Please install vLLM or use --backend transformers")
            sys.exit(1)

    def _init_transformers_backend(self):
        """Initialize Transformers backend"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            print("Initializing Transformers model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            print("Transformers backend initialized successfully")

        except ImportError as e:
            print(f"Error: Transformers backend requires transformers to be installed: {e}")
            sys.exit(1)

    def process_batch_vllm(
        self,
        image_paths: List[str],
        prompt: str,
        output_dir: str,
        save_metadata: bool = True
    ) -> List[Dict]:
        """Process images in batch using vLLM backend"""
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        import re

        def process_single_image(image_path):
            """Preprocess single image"""
            image = Image.open(image_path).convert('RGB')
            cache_item = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": self.processor.tokenize_with_images(
                        images=[image],
                        bos=True,
                        eos=True,
                        cropping=self.mode_config.crop_mode
                    )
                },
            }
            return cache_item

        # Preprocess images
        print(f"Preprocessing {len(image_paths)} images...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, image_paths),
                total=len(image_paths),
                desc="Preprocessing"
            ))

        # Generate OCR results
        print("Running OCR inference...")
        outputs_list = self.llm.generate(
            batch_inputs,
            sampling_params=self.sampling_params
        )

        # Process and save results
        results = []
        os.makedirs(output_dir, exist_ok=True)

        for output, image_path in tqdm(
            zip(outputs_list, image_paths),
            total=len(image_paths),
            desc="Saving results"
        ):
            content = output.outputs[0].text
            base_name = Path(image_path).stem

            # Save raw output with detection tags
            raw_path = os.path.join(output_dir, f"{base_name}_raw.md")
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Clean and save processed output
            cleaned_content = self._clean_output(content)
            clean_path = os.path.join(output_dir, f"{base_name}.md")
            with open(clean_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            result = {
                "image_path": image_path,
                "raw_output_path": raw_path,
                "clean_output_path": clean_path,
                "content": cleaned_content,
            }
            results.append(result)

        # Save metadata
        if save_metadata:
            metadata_path = os.path.join(output_dir, "batch_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_images": len(image_paths),
                    "mode": self.mode_config.name,
                    "prompt": prompt,
                    "backend": self.backend,
                    "results": results,
                }, f, indent=2)

        return results

    def process_batch_transformers(
        self,
        image_paths: List[str],
        prompt: str,
        output_dir: str,
        save_metadata: bool = True
    ) -> List[Dict]:
        """Process images using Transformers backend"""
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for image_path in tqdm(image_paths, desc="Processing images"):
            base_name = Path(image_path).stem

            # Process image
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=self.mode_config.base_size,
                image_size=self.mode_config.image_size,
                crop_mode=self.mode_config.crop_mode,
                save_results=True,
                test_compress=False
            )

            # The model.infer saves results automatically
            result_info = {
                "image_path": image_path,
                "output_dir": output_dir,
            }
            results.append(result_info)

        # Save metadata
        if save_metadata:
            metadata_path = os.path.join(output_dir, "batch_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_images": len(image_paths),
                    "mode": self.mode_config.name,
                    "prompt": prompt,
                    "backend": self.backend,
                    "results": results,
                }, f, indent=2)

        return results

    def _clean_output(self, text: str) -> str:
        """Clean OCR output by removing detection tags"""
        import re

        # Remove detection tags
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)

        cleaned = text
        for match in matches:
            cleaned = cleaned.replace(match[0], '')

        # Clean up extra newlines
        cleaned = cleaned.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
        cleaned = cleaned.replace('<center>', '').replace('</center>', '')

        return cleaned

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        extensions: List[str] = None,
        save_metadata: bool = True
    ) -> List[Dict]:
        """
        Process all images in a directory

        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results
            prompt: OCR prompt to use
            extensions: List of file extensions to process (default: jpg, jpeg, png)
            save_metadata: Whether to save processing metadata

        Returns:
            List of result dictionaries
        """
        if extensions is None:
            extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(input_dir, f"*.{ext.upper()}")))

        if not image_paths:
            print(f"No images found in {input_dir} with extensions: {extensions}")
            return []

        print(f"Found {len(image_paths)} images")
        print(f"Processing mode: {self.mode_config.description}")
        print(f"Backend: {self.backend}")

        # Initialize backend
        if self.backend == "vllm":
            if self.llm is None:
                self._init_vllm_backend()
            return self.process_batch_vllm(image_paths, prompt, output_dir, save_metadata)
        else:
            if self.model is None:
                self._init_transformers_backend()
            return self.process_batch_transformers(image_paths, prompt, output_dir, save_metadata)


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek OCR Batch Processing Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory with vLLM (fastest)
  python batch_ocr.py --input-dir ./images --output-dir ./results

  # Use transformers backend (simpler setup)
  python batch_ocr.py --input-dir ./images --output-dir ./results --backend transformers

  # Use specific processing mode
  python batch_ocr.py --input-dir ./images --output-dir ./results --mode base

  # Custom prompt for free OCR
  python batch_ocr.py --input-dir ./images --output-dir ./results --prompt "<image>\\nFree OCR."

Available modes:
  tiny   - 512x512, 64 vision tokens
  small  - 640x640, 100 vision tokens
  base   - 1024x1024, 256 vision tokens
  large  - 1280x1280, 400 vision tokens
  gundam - Dynamic resolution (recommended, default)
        """
    )

    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing images to process"
    )

    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to save OCR results"
    )

    parser.add_argument(
        "--model-path", "-m",
        default="deepseek-ai/DeepSeek-OCR",
        help="Path to DeepSeek OCR model (default: deepseek-ai/DeepSeek-OCR)"
    )

    parser.add_argument(
        "--backend", "-b",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Processing backend (default: vllm)"
    )

    parser.add_argument(
        "--mode",
        choices=["tiny", "small", "base", "large", "gundam"],
        default="gundam",
        help="Processing mode (default: gundam)"
    )

    parser.add_argument(
        "--prompt", "-p",
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="OCR prompt (default: Convert to markdown)"
    )

    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        default=["jpg", "jpeg", "png"],
        help="File extensions to process (default: jpg jpeg png)"
    )

    parser.add_argument(
        "--gpu-id",
        default="0",
        help="GPU device ID (default: 0)"
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="Max concurrent requests for vLLM (default: 100)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=64,
        help="Number of preprocessing workers (default: 64)"
    )

    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available processing modes and exit"
    )

    args = parser.parse_args()

    if args.list_modes:
        print("\nAvailable Processing Modes:")
        print("-" * 60)
        for mode in OCRModes.list_modes():
            print(f"{mode.name:8} - {mode.description}")
        return

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create OCR wrapper
    wrapper = DeepSeekOCRWrapper(
        model_path=args.model_path,
        backend=args.backend,
        mode=args.mode,
        gpu_id=args.gpu_id,
        max_concurrency=args.max_concurrency,
        num_workers=args.num_workers,
    )

    # Process directory
    print("\n" + "="*60)
    print("DeepSeek OCR Batch Processing")
    print("="*60)

    results = wrapper.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        extensions=args.extensions,
    )

    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Processed {len(results)} images")
    print(f"Results saved to: {args.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
