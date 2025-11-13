"""
script for evaluating Gemma models

"""
#pip install -U transformers
import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoModelForImageTextToText
from huggingface_hub import login


def generate_prompts(qa_dataset, outcome, image_dir):
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

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a critical care physician analyzing ICU patient data and chest X-rays. "
                                "Provide precise only yes/no answers based on the given information."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ]

            if qa_pair.get("image"):
                image_path = os.path.join(image_dir, os.path.basename(qa_pair["image"]))
                if os.path.isfile(image_path):
                    image = Image.open(image_path).convert("RGB").resize((512, 512))
                    messages[1]["content"].insert(0, {"type": "image", "image": image})

            prompts.append(messages)
            pbar.update(1)

    return prompts


def build_model_and_processor(model_id):
    """Load Gemma or MedGemma model and its processor."""

    if model_id == "google/gemma-3-4b-it":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()
    elif model_id == "google/medgemma-4b-it":
        # Use bf16 on CUDA; default dtype on CPU for compatibility
        if torch.cuda.is_available():
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, device_map="auto"
            )

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_inference(model,model_id,processor,prompts,max_new_tokens,
                  output_dir,outcome,progress_save_every):
    """Run generation over prompts, saving periodic partial results."""

    model_mapping = {"google/gemma-3-4b-it":"gemma4b","google/medgemma-4b-it":"medgemma4b"}
    os.makedirs(output_dir, exist_ok=True)
    answers = []
    nfiles = len(prompts)

    use_bf16 = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    with tqdm(total=nfiles) as pbar:
        for prompt in prompts:
            try:
                inputs = processor.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device, dtype=dtype)

                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False # Deterministic output for yes/no
                    )
                # Extract only the generated tokens (trim input)
                generated_tokens = generated_ids[0][input_len:]

                # Decode
                output_text = processor.decode(
                    generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                answers.append(output_text)
            except Exception as e:
                print(f"Error processing prompt at index {pbar.n}: {e}")
                answers.append(f"Error: {e}")

            pbar.update(1)
            if pbar.n % progress_save_every == 0 or pbar.n == nfiles:
                partial_path = os.path.join(output_dir, f"{model_mapping[model_id]}_{outcome}_output_{pbar.n}.npz")
                np.savez(partial_path, array=np.array(answers))
                print(f"Saved checkpoint: {partial_path}")

    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Gemma-3 4B-IT over QA dataset")
    parser.add_argument("--data-dir", default="Data/", help="Directory containing dataset and images")
    parser.add_argument("--json-name", default="qa_dataset_test.json", help="Dataset JSON filename")
    parser.add_argument("--image-dir", default="Data/jpg_test/", help="Directory for images")
    parser.add_argument("--output-dir", default="Outputs/", help="Directory for output files")
    parser.add_argument("--outcome", default="sepsis3", help="Outcome key in question field to target")
    parser.add_argument("--model-id", default="google/gemma-3-4b-it", help="HuggingFace model id")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--progress-save-every", type=int, default=2000, help="Save partial results every N prompts")
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, args.json_name)
    model_mapping = {"google/gemma-3-4b-it":"gemma4b","google/medgemma-4b-it":"medgemma4b"}

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    # Login to Hugging Face
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=True)
        except Exception as e:
            print(f"Warning: Hugging Face login failed: {e}")

    with open(json_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    prompts = generate_prompts(qa_dataset, args.outcome, args.image_dir)
    model, processor = build_model_and_processor(args.model_id)
    answers = run_inference(
        model,
        args.model_id,
        processor,
        prompts,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        outcome=args.outcome,
        progress_save_every=args.progress_save_every,
    )

    full_path = os.path.join(args.output_dir, f"{model_mapping[args.model_id]}_{args.outcome}_output.npz")
    np.savez(full_path, array=np.array(answers))
    print(f"Full results saved to {full_path}")


if __name__ == "__main__":
    main()
