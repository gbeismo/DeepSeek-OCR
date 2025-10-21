#!/usr/bin/env python3
"""
Example: Using Different Prompts

This example demonstrates using different prompts for different OCR tasks.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_ocr import DeepSeekOCRWrapper


def process_with_prompt(prompt_name, prompt, input_dir, output_dir):
    """Process images with a specific prompt"""
    print(f"\n{'='*60}")
    print(f"Processing with: {prompt_name}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    wrapper = DeepSeekOCRWrapper(
        backend="vllm",
        mode="gundam",
        gpu_id="0"
    )

    results = wrapper.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        prompt=prompt
    )

    print(f"Processed {len(results)} images")
    return results


def main():
    input_dir = "./input_images"
    base_output_dir = "./prompt_examples"

    # Define different prompts for different use cases
    prompts = {
        "markdown_conversion": {
            "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
            "description": "Best for documents, preserves layout and structure"
        },
        "free_ocr": {
            "prompt": "<image>\nFree OCR.",
            "description": "Simple text extraction without layout"
        },
        "parse_figure": {
            "prompt": "<image>\nParse the figure.",
            "description": "For charts, graphs, and figures"
        },
        "detailed_description": {
            "prompt": "<image>\nDescribe this image in detail.",
            "description": "General image understanding"
        },
        "ocr_image": {
            "prompt": "<image>\n<|grounding|>OCR this image.",
            "description": "OCR for general images (not documents)"
        }
    }

    # Process with each prompt
    for prompt_name, config in prompts.items():
        output_dir = os.path.join(base_output_dir, prompt_name)

        print(f"\n{config['description']}")

        process_with_prompt(
            prompt_name=prompt_name,
            prompt=config["prompt"],
            input_dir=input_dir,
            output_dir=output_dir
        )

    print(f"\n{'='*60}")
    print("All prompts processed!")
    print(f"Results saved to: {base_output_dir}")
    print(f"{'='*60}")
    print("\nCompare the results in each subdirectory:")
    for prompt_name, config in prompts.items():
        print(f"  {prompt_name}/ - {config['description']}")


if __name__ == "__main__":
    if not os.path.exists("./input_images"):
        print("Error: ./input_images directory not found")
        print("Please create it and add some images first")
        sys.exit(1)

    main()
