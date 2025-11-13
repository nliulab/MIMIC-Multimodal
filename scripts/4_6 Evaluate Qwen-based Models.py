"""
script for evaluating Qwen2.5-VL-7B and Lingshu

"""
# pip install git+https://github.com/huggingface/transformers accelerate
# pip install -U bitsandbytes
# pip install qwen-vl-utils[decord]==0.0.8
# pip install ninja
# pip install flash-attn --no-build-isolation

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


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

            image_path = None
            if qa_pair.get("image") is not None:
                p = os.path.join(image_dir, os.path.basename(qa_pair["image"]))
                if os.path.isfile(p):
                    image_path = p

            if image_path is not None:
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_text},
                        ],
                    }
                ]
            else:
                prompt = [
                    {"role": "user", "content": [{"type": "text", "text": user_text}]}
                ]

            prompts.append(prompt)
            pbar.update(1)
    return prompts


def build_model_and_processor(model_id,use_4bit = True):
    """Load Qwen2.5-VL based models and processor"""
    quantization_config = None
    if use_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    if model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
        )
    elif model_id == "lingshu-medical-mllm/Lingshu-7B":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def run_inference(model,model_id,processor,prompts,max_new_tokens,
                  output_dir,outcome,progress_save_every):
    """Run generation over prompts, saving periodic partial results."""
    model_mapping = {"Qwen/Qwen2.5-VL-7B-Instruct":"qwen2.5vl",
                     "lingshu-medical-mllm/Lingshu-7B":"lingshu"}

    os.makedirs(output_dir, exist_ok=True)
    answers = []
    nfiles = len(prompts)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tqdm(total=nfiles) as pbar:
        for prompt in prompts:
            # Prepare inputs
            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(prompt)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(device)

            # Generate and decode new tokens only
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Append
            answers.append(output_text)

            pbar.update(1)
            if pbar.n % progress_save_every == 0 or pbar.n == nfiles:
                partial_path = os.path.join(output_dir, f"{model_mapping[model_id]}_{outcome}_output_{pbar.n}.npz")
                np.savez(partial_path, array=np.array(answers))
                print(f"Saved checkpoint: {partial_path}")

    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL based models over QA dataset")
    parser.add_argument("--data-dir", default="Data/", help="Directory containing dataset and images")
    parser.add_argument("--json-name", default="qa_dataset_test.json", help="Dataset JSON filename")
    parser.add_argument("--image-dir", default="Data/jpg_test/", help="Directory for images")
    parser.add_argument("--output-dir", default="Outputs/", help="Directory for output files")
    parser.add_argument("--outcome", default="sepsis3", help="Outcome key in question field to target")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HuggingFace model id")
    parser.add_argument("--no-quant4bit", action="store_true", help="Disable 4-bit quantization (CPU or full precision)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--progress-save-every", type=int, default=2000, help="Save partial results every N prompts")
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, args.json_name)
    model_mapping = {"Qwen/Qwen2.5-VL-7B-Instruct":"qwen2.5vl",
                     "lingshu-medical-mllm/Lingshu-7B":"lingshu"}

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    prompts = generate_prompts(qa_dataset, args.outcome, args.image_dir)
    model, processor = build_model_and_processor(args.model_id, use_4bit=not args.no_quant4bit)
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