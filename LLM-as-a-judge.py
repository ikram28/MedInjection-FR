import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub

huggingface_hub.login('')

torch.cuda.empty_cache()

def construct_prompt(question: str, ca: str, gt: str) -> List[Dict[str, str]]:
    base_prompt = """You are a medical evaluator tasked with assessing whether a candidate answer is equivalent to the ground truth.
Assign a score strictly according to the criteria below. Do not include any explanations, comments, or extra text in your response.

Scoring criteria:
Score 0: Not equivalent
Score 1: Equivalent

Definition of equivalence:
Two answers are considered equivalent if the essential expected information is covered. Minor differences in wording, or additional or missing details, are acceptable as long as the candidate answer would be considered an acceptable response to the question.

Return only the score (0 or 1), nothing else."""


    return [
        {"role": "system", "content": base_prompt},
        {"role": "user",
         "content": f"Question: {question.strip()}\nCandidate_answer: {ca.strip()}\nGround_truth: {gt.strip()}"},
    ]


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return torch.bfloat16 if (major, minor) >= (8, 0) else torch.float16
    return torch.float32


def find_first(messages: List[Dict[str, Any]], role: str) -> Optional[str]:
    for m in messages or []:
        if m.get("role") == role:
            return m.get("content")
    return None


def extract_triplet(ex: Dict[str, Any]) -> Dict[str, Optional[str]]:
    msgs = ex.get("messages") or []
    q = find_first(msgs, "user")
    gt = find_first(msgs, "assistant")
    ca = ex.get("greedy_output")  # may be None
    return {"question": q, "gt": gt, "ca": (ca or "")}


def parse_score(text: str) -> Optional[int]:
    import re
    m = re.search(r"\b[01]\b", text) or re.search(r"[01]", text)
    return int(m.group(0)) if m else None


def build_inputs_for_batch(
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    device: torch.device
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
   
    # Render chat to strings once, then tokenize as a batch for speed
    rendered = [
        tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False  
        )
        for msgs in messages_list
    ]
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=False
    )

    
    attention_mask = enc["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).tolist()

    
    for k in ("input_ids", "attention_mask"):
       enc[k] = enc[k].to(device)

    return enc, input_lengths


def batch_generate_scores(
    model,
    tokenizer,
    messages_batch: List[List[Dict[str, str]]],
    max_new_tokens: int,
    temperature: float
) -> Tuple[List[str], List[Optional[int]]]:
  
    enc, input_lengths = build_inputs_for_batch(tokenizer, messages_batch, model.device)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    
    conts = []
    for i, in_len in enumerate(input_lengths):
        conts.append(out[i, in_len:])

    
    max_len = max(t.size(0) for t in conts) if conts else 0
    if max_len == 0:
        return [], []
    padded = torch.full((len(conts), max_len), fill_value=tokenizer.pad_token_id, dtype=conts[0].dtype, device=conts[0].device)
    for i, t in enumerate(conts):
        padded[i, :t.size(0)] = t

    decoded_batch = tokenizer.batch_decode(padded, skip_special_tokens=True)
    decoded_batch = [s.strip() for s in decoded_batch]
    scores = [parse_score(s) for s in decoded_batch]
    return decoded_batch, scores


def process_file(path: str,
                 model,
                 tokenizer,
                 max_new_tokens: int,
                 temperature: float,
                 skip_if_present: bool,
                 batch_size: int) -> Tuple[int, str]:
 
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Warn] Skipping {path}: cannot load JSON ({e})", file=sys.stderr)
        return 0, ""

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        print(f"[Warn] Skipping {path}: top-level JSON is not a list/dict", file=sys.stderr)
        return 0, ""

    # Collect indices to score
    todo_indices = []
    messages_buffer: List[List[Dict[str, str]]] = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            print(f"[Warn] {path}[{i}]: not an object, skipping", file=sys.stderr)
            continue
        if skip_if_present and ("eval_raw_output" in ex and "eval_score" in ex):
            continue

        mapping = extract_triplet(ex)
        q, gt, ca = mapping["question"], mapping["gt"], mapping["ca"]
        if not q or not gt:
            print(f"[Warn] {path}[{i}]: missing question/gt, skipping", file=sys.stderr)
            continue

        msgs = construct_prompt(q, ca, gt)
        todo_indices.append(i)
        messages_buffer.append(msgs)

    updated = 0
   
    for start in range(0, len(todo_indices), batch_size):
        end = min(start + batch_size, len(todo_indices))
        batch_msgs = messages_buffer[start:end]
        raw_batch, score_batch = batch_generate_scores(
            model, tokenizer, batch_msgs, max_new_tokens, temperature
        )
        
        for local_idx, (raw, score) in enumerate(zip(raw_batch, score_batch)):
            ex_idx = todo_indices[start + local_idx]
            data[ex_idx]["eval_raw_output"] = raw
            data[ex_idx]["eval_score"] = score
            updated += 1

    
    base, ext = os.path.splitext(path)
    out_path = f"{base}_scored{ext or '.json'}"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Error] Failed to write {out_path}: {e}", file=sys.stderr)
        return 0, ""

    return updated, out_path


def find_json_files(root_dir: str) -> List[str]:
    files = []
    # Expected structure: root_dir/<model_name>/*.json
    for model_dir in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, model_dir)
        if not os.path.isdir(full):
            continue
        for name in sorted(os.listdir(full)):
            if name.lower().endswith(".json"):
                files.append(os.path.join(full, name))
    return files


def parse_max_memory_arg(max_memory_arg: Optional[str], gpus: List[int]) -> Optional[Dict[int, str]]:
    
    if not max_memory_arg:
        return None
    parts = [p.strip() for p in max_memory_arg.split(",") if p.strip()]
    if len(parts) != len(gpus):
        print(f"[Warn] --max-memory entries ({len(parts)}) != number of GPUs ({len(gpus)}); ignoring.", file=sys.stderr)
        return None
    return {gpu: mem for gpu, mem in zip(gpus, parts)}



def main():
    parser = argparse.ArgumentParser(description="Evaluate candidate answers across a root dir and write results to NEW files.")
    parser.add_argument("--model-id", type=str, default="google/medgemma-27b-text-it")
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Root directory containing one subdirectory per model; each subdir has JSON result files.")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", type=str, default="auto",
                        help='Transformers device_map')
    parser.add_argument("--gpus", type=str, default="",
                        help="Comma-separated GPU ids to use. If empty, use all visible GPUs.")
    parser.add_argument("--max-memory", type=str, default="",
                        help="Comma-separated per-GPU memory limits. Optional.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-if-present", action="store_true",
                        help="If set, skip items that already have eval_raw_output & eval_score.")
    args = parser.parse_args()

    # Seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    
    visible_gpus = list(range(torch.cuda.device_count()))
    use_gpus = visible_gpus if not args.gpus else [int(x) for x in args.gpus.split(",") if x.strip().isdigit()]
    if not use_gpus:
        use_gpus = visible_gpus

    max_memory = parse_max_memory_arg(args.max_memory, use_gpus)

    
    dtype = pick_dtype()
    print(
        f"[Info] Loading model {args.model_id} (dtype={dtype}, device_map={args.device_map}, gpus={use_gpus}, max_memory={max_memory})",
        file=sys.stderr
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,    
        max_memory=max_memory,         
        low_cpu_mem_usage=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    
    files = find_json_files(args.root_dir)
    if not files:
        print(f"[Error] No JSON files found under {args.root_dir}", file=sys.stderr)
        sys.exit(1)

    total_items = 0
    total_files = 0
    for path in files:
        updated, out_path = process_file(
            path, model, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            skip_if_present=args.skip_if_present,
            batch_size=args.batch_size,
        )
        rel_in = os.path.relpath(path, args.root_dir)
        rel_out = os.path.relpath(out_path, args.root_dir) if out_path else "(no output)"
        print(f"[Info] {rel_in} -> {rel_out}: updated {updated} items", file=sys.stderr)
        total_items += updated
        total_files += 1

    print(f"[Done] Processed {total_files} files. Updated {total_items} items.", file=sys.stderr)


if __name__ == "__main__":
    main()
