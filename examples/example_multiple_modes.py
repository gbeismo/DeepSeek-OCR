#!/usr/bin/env python3
"""
Example: Compare Different Processing Modes

This example processes the same images with different modes to compare quality/speed.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_ocr import DeepSeekOCRWrapper, OCRModes


def process_with_mode(mode_name, input_dir, base_output_dir):
    """Process images with a specific mode"""
    print(f"\n{'='*60}")
    print(f"Processing with {mode_name.upper()} mode")
    print(f"{'='*60}")

    output_dir = os.path.join(base_output_dir, mode_name)

    wrapper = DeepSeekOCRWrapper(
        backend="vllm",
        mode=mode_name,
        gpu_id="0"
    )

    start_time = time.time()

    results = wrapper.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        prompt="<image>\n<|grounding|>Convert the document to markdown."
    )

    elapsed_time = time.time() - start_time

    print(f"Completed in {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(results):.2f} seconds")

    return {
        "mode": mode_name,
        "time": elapsed_time,
        "count": len(results),
        "avg_time": elapsed_time / len(results) if results else 0
    }


def main():
    input_dir = "./input_images"
    output_dir = "./mode_comparison"

    # Test different modes
    modes_to_test = ["tiny", "small", "base", "gundam"]

    results = []
    for mode in modes_to_test:
        result = process_with_mode(mode, input_dir, output_dir)
        results.append(result)

    # Print comparison
    print(f"\n{'='*60}")
    print("MODE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<10} {'Total Time':>12} {'Avg Time/Image':>18}")
    print("-" * 60)

    for result in results:
        print(f"{result['mode']:<10} {result['time']:>10.2f}s {result['avg_time']:>16.2f}s")

    print(f"\nResults saved to: {output_dir}")
    print("Compare the output quality in each subdirectory!")


if __name__ == "__main__":
    if not os.path.exists("./input_images"):
        print("Error: ./input_images directory not found")
        print("Please create it and add some images first")
        sys.exit(1)

    main()
