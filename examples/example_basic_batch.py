#!/usr/bin/env python3
"""
Example: Basic Batch Processing

This example shows the simplest way to batch process images with DeepSeek OCR.
"""

import sys
import os

# Add parent directory to path to import batch_ocr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_ocr import DeepSeekOCRWrapper


def main():
    # Create OCR wrapper with default settings
    # By default uses vLLM backend and Gundam mode
    wrapper = DeepSeekOCRWrapper(
        backend="vllm",  # or "transformers"
        mode="gundam",   # or "tiny", "small", "base", "large"
        gpu_id="0"
    )

    # Process all images in a directory
    results = wrapper.process_directory(
        input_dir="./input_images",
        output_dir="./ocr_results",
        prompt="<image>\n<|grounding|>Convert the document to markdown."
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")

    # Show first few results
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. {os.path.basename(result['image_path'])}")
        print(f"   Output: {result['clean_output_path']}")
        print(f"   Preview: {result['content'][:100]}...")


if __name__ == "__main__":
    # Check if input directory exists
    if not os.path.exists("./input_images"):
        print("Error: ./input_images directory not found")
        print("Please create ./input_images and add some images")
        print("\nExample:")
        print("  mkdir input_images")
        print("  cp /path/to/your/images/*.jpg input_images/")
        sys.exit(1)

    main()
