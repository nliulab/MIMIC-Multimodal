"""
script for evaluating GPT4o-mini
"""

import argparse
import base64
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

def encode_image(image_path: str) -> str:
    """Return base64 string for an image file."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_prompts(qa_dataset, outcome, image_dir):
    prompts = []

    nfiles = len(qa_dataset)
    # Generate user prompts for all QA pairs
    with tqdm(total=nfiles) as pbar:
        for i in range(nfiles):
            prompt = []
            qa_pair = qa_dataset[i]
            # Text
            user_text = (
                f"Based on the information collected during current ICU stay, \n{qa_pair['context']}\n"
                f"{qa_pair['question'][outcome]}\n"
                f"Answer the question using only yes or no."
                )
            prompt.append({"type": "text", "text": user_text})

            # Images
            if qa_pair['image'] != None:
                path = os.path.join(image_dir,os.path.basename(qa_pair['image']))
                encoded_image = encode_image(path)
                prompt.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "low"
                        }
                    })

            # Append
            prompts.append(prompt)
            # Update
            pbar.update(1)

    return prompts

def call_model_and_collect(client,model,system_prompt,user_prompts,output_dir,outcome,progress_save_every=2000):
    """Call the model for each prompt and return answers; periodically save progress."""
    os.makedirs(output_dir, exist_ok=True)
    answers = []

    nfiles = len(user_prompts)
    with tqdm(total=nfiles) as pbar:
        for user_prompt in user_prompts:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.choices[0].message.content
            answers.append(answer)

            pbar.update(1)
            if pbar.n % progress_save_every == 0 or pbar.n == nfiles:
                partial_path = os.path.join(
                    output_dir, f"gpt_{outcome}_output_{pbar.n}.npz"
                )
                np.savez(partial_path, array=np.array(answers))
                print(f"Saved checkpoint: {partial_path}")

    return np.array(answers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o-mini over QA dataset")
    parser.add_argument(
        "--data-dir",
        default="Data/",
        help="Directory containing dataset and images",
    )
    parser.add_argument(
        "--json-name",
        default="qa_dataset_test.json",
        help="JSON filename for QA dataset",
    )
    parser.add_argument(
        "--image-dir",
        default="Data/jpg_test/",
        help="Directory for images",
    )
    parser.add_argument(
        "--output-dir",
        default="Outputs/",
        help="Directory for periodic partial outputs",
    )
    parser.add_argument(
        "--outcome",
        default="sepsis3",
        help="Outcome key in question field to target",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model identifier",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful critical care physician.",
        help="System prompt for the assistant",
    )
    parser.add_argument(
        "--progress-save-every",
        type=int,
        default=2000,
        help="Save partial results every N prompts",
    )
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, args.json_name)

    # export OPENAI_API_KEY=<your_api_key>
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Please export your API key.")

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    # Load dataset
    with open(json_path, "r", encoding="utf-8") as f:
        qa_dataset_test = json.load(f)

    # Build prompts
    user_prompts = generate_prompts(qa_dataset_test, args.outcome, args.image_dir)

    # Initialize client and run inference
    client = OpenAI(api_key=api_key)
    answers = call_model_and_collect(
        client=client,
        model=args.model,
        system_prompt=args.system_prompt,
        user_prompts=user_prompts,
        output_dir=args.output_dir,
        outcome = args.outcome,
        progress_save_every=args.progress_save_every,
    )

    full_path = os.path.join(args.output_dir, f"gpt_{args.outcome}_output.npz")
    np.savez(full_path, array=np.array(answers))
    print(f"Full results saved to {full_path}")


if __name__ == "__main__":
    main()
