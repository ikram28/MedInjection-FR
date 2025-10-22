"""
Evaluate Qwen on MCQ and OEQA datasets

Features:
- Deterministic greedy decoding for all, plus constrained decoding for MCQs.
- Constrained decoding masks vocab to class letters (+ separators, EOS/EOT).
"""

from __future__ import annotations
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from huggingface_hub import login
login(token="")


# ------------------------- Utilities -------------------------

def load_json_items(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    else:
        return [data]

def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def dataset_kind_from_filename(filename: str) -> str:
    name = filename.lower()
    if "qcmu" in name:
        return "qcmu"
    if "qcm" in name:
        return "qcm"
    if any(tag in name for tag in ["qro", "en2fr", "fr2en"]):
        return "oeqa"
    return "unknown"

def filter_messages_for_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [m for m in messages if m.get("role") != "assistant"]

def apply_chat(tokenizer, messages: List[Dict[str, str]]) -> torch.LongTensor:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

def token_ids_for_strings(tokenizer, strings: List[str]) -> List[int]:
    ids = []
    for s in strings:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if enc:
            ids.append(enc[-1]) 
    return sorted(set(ids))


class AllowedTokens(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        super().__init__()
        self.allowed = torch.tensor(sorted(set(allowed_token_ids)), dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        mask = torch.full((vocab_size,), float("-inf"), device=scores.device)
        mask[self.allowed.to(scores.device)] = 0.0
        scores = scores + mask
        return scores


# ------------------------- Parsing & metrics -------------------------

def parse_letters_from_text(text: str, valid_letters: List[str], multi: bool) -> List[str]:
    if text is None:
        return []
    s = text.upper().strip()
    letters = [v.upper() for v in valid_letters]
    letter_set = set(letters)
    allowed_nonletters = set([',', ';', '/', '|', '+', ' ', '\n', '.', '(', ')', ':'])

    for ch in s:
        if ch.isalpha() and ch not in letter_set:
            return []
        if not ch.isalpha() and ch not in allowed_nonletters:
            return []

    preds = [ch for ch in s if ch in letter_set]
    seen = set()
    preds = [x for x in preds if not (x in seen or seen.add(x))]
    if not multi:
        return preds if len(preds) == 1 else []
    return preds

def normalize_mcq_answer(ans: str | List[str]) -> List[str]:
    if isinstance(ans, list):
        items = ans
    else:
        items = re.split(r"[\s,]+", ans)
    return [a.strip().upper() for a in items if a.strip()]

def em_score(true_labels: List[str], pred_labels: List[str]) -> float:
    return 1.0 if set(true_labels) == set(pred_labels) else 0.0

def hamming_score(true_labels: List[str], pred_labels: List[str]) -> float:
    set_t = set(true_labels)
    set_p = set(pred_labels)
    if not set_t and not set_p:
        return 1.0
    union = len(set_t | set_p)
    inter = len(set_t & set_p)
    return inter / union if union > 0 else 0.0


# ------------------------- Token→letter helpers -------------------------

def build_letter_token_mapping(tokenizer, letters: List[str]):
    letter_token_ids = token_ids_for_strings(tokenizer, letters + [" " + l for l in letters])
    tokid_to_letter: Dict[int, str] = {}
    for L in letters:
        for s in [L, " " + L]:
            enc = tokenizer.encode(s, add_special_tokens=False)
            if enc:
                tokid_to_letter[enc[-1]] = L
    return letter_token_ids, tokid_to_letter

def step_probs_from_scores_list(
    scores_list: List[torch.Tensor],
    tokid_to_letter: Dict[int, str],
    letter_token_ids: List[int]
) -> List[Dict[str, float]]:
    if not scores_list or not letter_token_ids:
        return []
    letter_ids_tensor = torch.tensor(letter_token_ids, dtype=torch.long, device=scores_list[0].device)
    out: List[Dict[str, float]] = []
    for step_scores in scores_list:  # [batch, vocab]
        vec = step_scores[0]
        sel = vec[letter_ids_tensor]
        probs = torch.softmax(sel, dim=-1)
        d: Dict[str, float] = {}
        for tid, p in zip(letter_ids_tensor.tolist(), probs.tolist()):
            L = tokid_to_letter.get(tid)
            if L is None:
                continue
            d[L] = d.get(L, 0.0) + float(p)
        out.append(d)
    return out


# ------------------------- Core evaluation -------------------------

def evaluate_file(
    model, tokenizer, path: str, out_dir: str, device: str
) -> Tuple[str, Dict[str, float]]:
    kind = dataset_kind_from_filename(os.path.basename(path))
    items = load_json_items(path)
    results: List[Dict[str, Any]] = []

    eos_id = tokenizer.eos_token_id

    n = 0
    em_sum_greedy = 0.0
    em_sum_constr = 0.0
    ham_sum_greedy = 0.0
    ham_sum_constr = 0.0

    for ex in items:
        n += 1
        messages = filter_messages_for_prompt(ex.get("messages", []))
        prompt_ids = apply_chat(tokenizer, messages).to(device)

        if kind == "qcmu":
            max_new = 1
        elif kind == "qcm":
            max_new = max(2 * len(ex.get("classes", [])), 1)
        else:
            max_new = 512 if kind == "oeqa" else 128

        # Greedy unconstrained
        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=max_new,
                eos_token_id=eos_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0, prompt_ids.shape[1]:]
        greedy_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        greedy_step_class_probabilities: List[Dict[str, float]] = []

        constrained_text = None
        class_probabilities: Optional[Dict[str, float]] = None
        constrained_step_class_probabilities: List[Dict[str, float]] = []
        pred_constrained: Optional[List[str]] = None
        greedy_parsed: Optional[List[str]] = None
        pred_greedy: Optional[List[str]] = None

        classes = ex.get("classes") if isinstance(ex.get("classes"), list) else None

        if classes and kind in {"qcmu", "qcm"}:
            letters = [c.strip().upper() for c in classes]

            letter_ids = token_ids_for_strings(tokenizer, letters + [" " + l for l in letters])
            sep_ids = token_ids_for_strings(tokenizer, [",", ", ", " ", ",\n"])
            stop_ids: List[int] = []
            if eos_id is not None:
                stop_ids.append(eos_id)
            try:
                for k, v in (tokenizer.special_tokens_map or {}).items():
                    if isinstance(v, str) and any(tag in v.lower() for tag in ["eos", "eot", "im_end"]):
                        tid = tokenizer.convert_tokens_to_ids(v)
                        if isinstance(tid, int) and tid >= 0:
                            stop_ids.append(tid)
            except Exception:
                pass

            allowed_for_gen = sorted(set(letter_ids + sep_ids + stop_ids))
            id2label = {tid: label_token(tokenizer, tid, eos_id) for tid in allowed_for_gen}


            # Constrained decoding
            with torch.no_grad():
                out_c = model.generate(
                    prompt_ids,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=max_new,
                    eos_token_id=eos_id,
                    logits_processor=[AllowedTokens(allowed_for_gen)],
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            gen_ids_c = out_c.sequences[0, prompt_ids.shape[1]:]
            constrained_text = tokenizer.decode(gen_ids_c, skip_special_tokens=True).strip()

            # Probabilities & per-step traces
            letter_ids_all, tokid_to_letter = build_letter_token_mapping(tokenizer, letters)
            greedy_step_class_probabilities = step_probs_from_scores_list(out.scores, tokid_to_letter, letter_ids_all)
            constrained_step_class_probabilities = step_probs_from_scores_list(out_c.scores, tokid_to_letter, letter_ids_all)


            # Per-step probabilities over ALL allowed tokens (letters + separators + EOS/EOT)
            greedy_step_allowed_token_probabilities = step_probs_over_allowed(out.scores, allowed_for_gen, id2label)
            constrained_step_allowed_token_probabilities = step_probs_over_allowed(out_c.scores, allowed_for_gen, id2label)

            # First-step snapshot over allowed tokens (for convenience)
            first_step_allowed_token_probabilities = (
                constrained_step_allowed_token_probabilities[0]
                if constrained_step_allowed_token_probabilities
                else (greedy_step_allowed_token_probabilities[0] if greedy_step_allowed_token_probabilities else None)
            )


            if constrained_step_class_probabilities:
                class_probabilities = constrained_step_class_probabilities[0]
            elif greedy_step_class_probabilities:
                class_probabilities = greedy_step_class_probabilities[0]
            else:
                class_probabilities = None

            # Predictions
            greedy_parsed = parse_letters_from_text(greedy_text, letters, multi=(kind == "qcm"))
            pred_greedy = greedy_parsed if greedy_parsed else []

            if kind == "qcmu":
                pred_constrained = parse_letters_from_text(constrained_text or "", letters, multi=False)
            else:
                pred_constrained = parse_letters_from_text(constrained_text or "", letters, multi=True)


        # Record
        rec = dict(ex)
        rec.update({
            "greedy_output": greedy_text,
            "greedy_output_parsed": greedy_parsed,
            "prediction_greedy": pred_greedy,
        })
        if kind in {"qcmu", "qcm"}:
            rec.update({
                "constrained_output": constrained_text,
                "class_probabilities": class_probabilities,
                "prediction_constrained": pred_constrained,
                "first_step_allowed_token_probabilities": first_step_allowed_token_probabilities,
                "greedy_step_allowed_token_probabilities": greedy_step_allowed_token_probabilities,
                "constrained_step_allowed_token_probabilities": constrained_step_allowed_token_probabilities,
            })
        results.append(rec)

        # Metrics
        if kind in {"qcmu", "qcm"}:
            gold = normalize_mcq_answer(ex.get("answer", ""))

            if pred_greedy is not None:
                em_sum_greedy += em_score(gold, pred_greedy)
                if kind == "qcm":
                    ham_sum_greedy += hamming_score(gold, pred_greedy)

            if pred_constrained is not None:
                em_sum_constr += em_score(gold, pred_constrained)
                if kind == "qcm":
                    ham_sum_constr += hamming_score(gold, pred_constrained)

    # Write per-file JSON
    base = os.path.basename(path)
    out_path = os.path.join(out_dir, os.path.splitext(base)[0] + ".qwen_eval.json")
    os.makedirs(out_dir, exist_ok=True)
    save_json(out_path, results)

    # Summary
    summary: Dict[str, float] = {"n_items": float(n)}
    if n > 0 and kind in {"qcmu", "qcm"}:
        if kind == "qcm":
            summary.update({
                "em_greedy": em_sum_greedy / n,
                "hamming_greedy": ham_sum_greedy / n,
                "em_constrained": em_sum_constr / n,
                "hamming_constrained": ham_sum_constr / n,
            })
        else:
            summary.update({
                "em_greedy": em_sum_greedy / n,
                "em_constrained": em_sum_constr / n,
            })

    return out_path, summary


def label_token(tokenizer, tid: int, eos_id: Optional[int]) -> str:
    """Human-readable label for a single token id."""
    if eos_id is not None and tid == eos_id:
        return "<eos>"
    
    s = tokenizer.decode([tid], skip_special_tokens=False)
    if s:
        return s
    # Fallback to raw token string if decode is empty
    try:
        tok = tokenizer.convert_ids_to_tokens(tid)
        return tok if tok is not None else f"<id:{tid}>"
    except Exception:
        return f"<id:{tid}>"


def step_probs_over_allowed(
    scores_list: List[torch.Tensor],
    allowed_token_ids: List[int],
    id2label: Dict[int, str],
) -> List[Dict[str, float]]:
    """
    For each generation step, softmax over *only* allowed_token_ids and
    return {label: prob} dict. Labels come from id2label.
    """
    if not scores_list or not allowed_token_ids:
        return []
    allowed_tensor = torch.tensor(allowed_token_ids, dtype=torch.long, device=scores_list[0].device)
    out: List[Dict[str, float]] = []
    for step_scores in scores_list:  # [batch, vocab]
        vec = step_scores[0]
        sel = vec[allowed_tensor]
        probs = torch.softmax(sel, dim=-1)
        d: Dict[str, float] = {}
        for tid, p in zip(allowed_tensor.tolist(), probs.tolist()):
            d[id2label.get(tid, f"<id:{tid}>")] = float(p)
        out.append(d)
    return out



def append_summary_csv(csv_path: str, dataset_name: str, kind: str, summary: Dict[str, float]):
    header = [
        "dataset", "type", "n_items", "em_greedy", "hamming_greedy", "em_constrained", "hamming_constrained",
    ]
    row = [
        dataset_name,
        kind,
        f"{int(summary.get('n_items', 0))}",
        f"{summary.get('em_greedy', float('nan')):.6f}" if "em_greedy" in summary else "",
        f"{summary.get('hamming_greedy', float('nan')):.6f}" if "hamming_greedy" in summary else "",
        f"{summary.get('em_constrained', float('nan')):.6f}" if "em_constrained" in summary else "",
        f"{summary.get('hamming_constrained', float('nan')):.6f}" if "hamming_constrained" in summary else "",
    ]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(row) + "\n")


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen on MCQ/OEQA datasets.")
    parser.add_argument("datasets_dir", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]

    print(f"Loading model {args.model} on {device} with dtype {torch_dtype}…")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval() 
    if device == "cpu":
        model = model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results_summary.csv")

    files = [f for f in os.listdir(args.datasets_dir) if f.lower().endswith(".json")]
    if not files:
        print("No JSON files found in", args.datasets_dir)
        return

    for fname in sorted(files):
        in_path = os.path.join(args.datasets_dir, fname)
        kind = dataset_kind_from_filename(fname)
        print(f"\nProcessing {fname} (type={kind})…")
        out_json_path, summary = evaluate_file(model, tokenizer, in_path, args.output_dir, device)
        print(f"  → Wrote {out_json_path}")
        append_summary_csv(csv_path, os.path.splitext(fname)[0], kind, summary)

    print(f"\nDone. Summary written to {csv_path}")


if __name__ == "__main__":
    main()
