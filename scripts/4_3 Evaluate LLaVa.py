"""
script for evaluating LLaVa

"""
# pip install --upgrade -q accelerate bitsandbytes
# pip install git+https://github.com/huggingface/transformers.git
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import pipeline, BitsAndBytesConfig



def generate_prompts(qa_dataset, outcome, image_dir):

    """Generate prompts with text and a PIL image per item."""
    prompts = []

    nfiles = len(qa_dataset)
    with tqdm(total=nfiles) as pbar:
        for i in range(nfiles):
            qa_pair = qa_dataset[i]
            user_text = (
                f"Based on the information collected during current ICU stay, \n{qa_pair['context']}\n"
                f"{qa_pair['question'][outcome]}\n"
                f"Answer the question using only yes or no."
            )

            # Image: use dataset image if present, else blank image
            image = None
            if qa_pair.get("image") is not None:
                image_path = os.path.join(image_dir, os.path.basename(qa_pair["image"]))
                if os.path.isfile(image_path):
                    image = Image.open(image_path).convert("RGB").resize((512, 512))
            if image is None:
                image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            prompts.append(prompt)
            pbar.update(1)

    return prompts


def build_pipeline(model_id, quant4bit=True):
    """Create a transformers pipeline for image-text-to-text with optional 4bit quantization."""
    model_kwargs = None
    if quant4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_kwargs = {"quantization_config": quantization_config}
    return pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs)


def run_inference(pipe,prompts,output_dir,outcome,max_new_tokens = 200,progress_save_every = 2000):
    """Run the pipeline over prompts, save periodic partials, and collect outputs."""
    ensure_dir(output_dir)
    answers = []
    nfiles = len(prompts)
    with tqdm(total=nfiles) as pbar:
        for prompt in prompts:
            output = pipe(text=prompt, max_new_tokens=max_new_tokens, return_full_text=False)
            answers.append(output[0]["generated_text"])
            pbar.update(1)

            if pbar.n % progress_save_every == 0 or pbar.n == nfiles:
                partial_path = os.path.join(output_dir, f"llava_{outcome}_output_{pbar.n}.npz")
                np.savez(partial_path, array=np.array(answers))
                print(f"Saved checkpoint: {partial_path}")
    return answers


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLaVa-style model over QA dataset")
    parser.add_argument("--data-dir", default="Data/", help="Directory containing dataset and images")
    parser.add_argument("--json-name", default="qa_dataset_test.json", help="Dataset JSON filename")
    parser.add_argument("--image-dir", default="Data/jpg_test/", help="Directory for images")
    parser.add_argument("--output-dir", default="Outputs/", help="Directory for output files")
    parser.add_argument("--outcome", default="sepsis3", help="Outcome key in question field to target")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="HuggingFace model id")
    parser.add_argument("--no-quant4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--progress-save-every", type=int, default=2000, help="Save partial results every N prompts")
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, args.json_name)
    image_dir = os.path.join(args.data_dir, args.image_dir)

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    prompts = generate_prompts(qa_dataset, args.outcome, image_dir)
    pipe = build_pipeline(args.model_id, quant4bit=not args.no_quant4bit)
    answers = run_inference(
        pipe,
        prompts,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        outcome=args.outcome,
        progress_save_every=args.progress_save_every,
    )

    full_path = os.path.join(args.output_dir, f"llava_{args.outcome}_output.npz")
    np.savez(full_path, array=np.array(answers))
    print(f"Full results saved to {full_path}")

if __name__ == "__main__":
    main()
