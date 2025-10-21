#!/usr/bin/env python3
"""
Example: Using Transformers Backend

This example shows how to use the Transformers backend instead of vLLM.
This is useful when vLLM is not available or for simpler setups.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_ocr import DeepSeekOCRWrapper


def main():
    print("Using Transformers Backend")
    print("="*60)
    print("Note: This is simpler but slower than vLLM for batch processing")
    print("Great for single images or when vLLM is not available")
    print("="*60)

    # Create wrapper with Transformers backend
    wrapper = DeepSeekOCRWrapper(
        backend="transformers",  # Use transformers instead of vLLM
        mode="base",             # Use base mode for good quality
        gpu_id="0"
    )

    # Process images
    results = wrapper.process_directory(
        input_dir="./input_images",
        output_dir="./transformers_output",
        prompt="<image>\n<|grounding|>Convert the document to markdown."
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Processed {len(results)} images")
    print(f"Results saved to: ./transformers_output")

    # Show results
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {os.path.basename(result['image_path'])}")
        print(f"   Output directory: {result['output_dir']}")


if __name__ == "__main__":
    if not os.path.exists("./input_images"):
        print("Error: ./input_images directory not found")
        print("Please create it and add some images first")
        sys.exit(1)

    main()
