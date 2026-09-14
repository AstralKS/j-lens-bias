#!/usr/bin/env python3
"""Week 1 Jacobian Lens batch experiment runner (Cohere Labs J-Lens Bias).

Standalone, sequential, checkpoint/resume-capable driver so the expensive
model computation doesn't need a live Claude/notebook session babysitting it.
Uses ONLY the released, pre-fitted Qwen3.5-0.8B Jacobian lens via the
official `jlens` package (read-only dependency: ~/Desktop/JLens-project/
jacobian-lens). Never calls jlens.fit().

Methodology (matches jlens.vis.compute_slice's slice-vis semantics exactly,
by reusing its private helpers rather than reimplementing them):
  - For each prompt, tokenize and take EVERY token position (0..seq_len-1)
    as the source position set.
  - Call lens.apply() ONCE per prompt with layers=lens.source_layers and
    positions=every source position -> one forward pass, full-vocab logits
    per (fitted layer, position), plus the model's own final-layer logits.
  - top1_token: argmax over the full-vocab logits restricted to "word-like"
    tokens (jlens.vis._meaningful_token_mask) -- the same masking the
    walkthrough's own rendered slice-vis pages use (mask_display=True).
  - rank: the FULL-VOCAB rank (0 = top) of that top1 token, computed with
    jlens.vis._ranks_of -- the exact function the slice-vis page uses for
    the rank superscript it displays.
  - The model's own final layer (index n_layers-1) is included as an extra
    row using model_logits (J = identity), matching how compute_slice
    renders the bottom "model output" row of the slice-vis grid.

Run:
    ~/Desktop/JLens-project/jacobian-lens/.venv/bin/python \
        "Week 1/scripts/run_week1_experiment.py" --n-prompts 3

Resumable: prompts with a valid existing checkpoint CSV are skipped. Safe to
re-run after an interruption; never deletes existing good checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
JACOBIAN_LENS_REPO = HOME / "Desktop/JLens-project/jacobian-lens"
SUBMISSION_REPO = HOME / "Desktop/JLens-project/j-lens-bias"
WEEK1_DIR = SUBMISSION_REPO / "Week 1"
RESULTS_DIR = WEEK1_DIR / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
COMPLETED_JSON = RESULTS_DIR / "week1_completed_prompts.json"
RAW_CSV = RESULTS_DIR / "week1_raw_results.csv"
MEMORY_LOG = RESULTS_DIR / "week1_memory_log.jsonl"

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"

# Diverse, official two-hop factual prompts from jacobian-lens's own
# data/experiments/probe-swap.json (90-item released prompt set). Picked to
# span distinct categories rather than clustering in "multihop".
PROBE_SWAP_JSON = JACOBIAN_LENS_REPO / "data/experiments/probe-swap.json"
SELECTED_NAMES = [
    "amazon-language",
    "animal-nose-elephant",
    "bird-country-eagle",
    "element-color-gold2",
    "organ-location-brain",
]

CSV_FIELDS = [
    "example_id",
    "example_name",
    "category",
    "prompt",
    "layer",
    "readout",
    "position",
    "source_token",
    "top1_token",
    "rank",
]

FREE_PCT_FLOOR = 20  # memory_pressure free% below this -> stop gracefully
SWAP_USED_MB_CEIL = 4800  # out of ~5120MB total swap on this Mac


def load_selected_prompts() -> list[dict]:
    data = json.loads(PROBE_SWAP_JSON.read_text())
    by_name = {it["name"]: it for it in data["items"]}
    prompts = []
    for i, name in enumerate(SELECTED_NAMES, start=1):
        it = by_name[name]
        prompts.append(
            {
                "example_id": f"example_{i:03d}",
                "example_name": name,
                "category": it["category"],
                "prompt": it["prompt"],
            }
        )
    return prompts


def check_memory() -> dict:
    """Lightweight macOS memory check via existing system tools only."""
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    free_pct = int(m.group(1)) if m else None

    swap_out = subprocess.run(
        ["sysctl", "vm.swapusage"], capture_output=True, text=True
    ).stdout
    m2 = re.search(r"used\s*=\s*([\d.]+)M", swap_out)
    swap_used_mb = float(m2.group(1)) if m2 else None

    return {
        "ts": time.time(),
        "free_pct": free_pct,
        "swap_used_mb": swap_used_mb,
    }


def log_memory(tag: str, snap: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_LOG, "a") as f:
        f.write(json.dumps({"tag": tag, **snap}) + "\n")
    print(
        f"[memory:{tag}] free={snap['free_pct']}% swap_used={snap['swap_used_mb']}MB",
        file=sys.stderr,
    )


def memory_is_safe(snap: dict) -> bool:
    if snap["free_pct"] is not None and snap["free_pct"] < FREE_PCT_FLOOR:
        return False
    if snap["swap_used_mb"] is not None and snap["swap_used_mb"] > SWAP_USED_MB_CEIL:
        return False
    return True


def checkpoint_path(example_id: str) -> Path:
    return CKPT_DIR / f"{example_id}.csv"


def checkpoint_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_FIELDS:
                return False
            return any(True for _ in reader)
    except Exception:
        return False


def write_checkpoint_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)  # atomic on same filesystem


def mark_completed(example_id: str) -> None:
    completed = []
    if COMPLETED_JSON.exists():
        completed = json.loads(COMPLETED_JSON.read_text())
    if example_id not in completed:
        completed.append(example_id)
    COMPLETED_JSON.write_text(json.dumps(completed, indent=2))


def top1_and_rank(logits, mask):
    """Official slice-vis top1/rank for a [n_positions, vocab] logits tensor:
    top1 = argmax restricted to word-like tokens (mask), rank = that token's
    FULL-VOCAB rank (0 = top), via jlens.vis._ranks_of. Shared by
    process_prompt() and the submission notebook's consistency check against
    jlens.vis.compute_slice() so both paths are guaranteed identical."""
    from jlens import vis

    masked = logits.masked_fill(~mask, float("-inf"))
    top1_ids = masked.argmax(dim=-1)  # [n_positions]
    ranks = vis._ranks_of(logits, top1_ids.unsqueeze(-1))[:, 0]  # [n_positions]
    return top1_ids, ranks


def process_prompt(model, tokenizer, lens, item: dict) -> list[dict]:
    """One forward pass; every source position x every lens layer + model
    output layer. Returns tidy row dicts. All heavy tensors stay local to
    this function so they go out of scope (and are gc'd) once it returns."""
    import torch
    from jlens import vis

    prompt = item["prompt"]
    input_ids = model.encode(prompt)
    seq_len = input_ids.shape[1]
    positions = list(range(seq_len))

    lens_logits, model_logits, _ = lens.apply(
        model, prompt, layers=lens.source_layers, positions=positions
    )
    # lens_logits: {layer: [seq_len, vocab]} on CPU float32 (apply() already
    # moves results off MPS). model_logits: [seq_len, vocab] on CPU float32.

    final_layer = model.n_layers - 1
    vocab_size = model_logits.shape[-1]
    mask = vis._meaningful_token_mask(tokenizer, vocab_size, model_logits.device)

    token_ids = input_ids[0].tolist()
    source_tokens = [
        tokenizer.decode([t], clean_up_tokenization_spaces=False) for t in token_ids
    ]

    rows: list[dict] = []

    def rows_for(layer_label: int, readout: str, logits: "torch.Tensor") -> None:
        top1_ids, ranks = top1_and_rank(logits, mask)
        for i, pos in enumerate(positions):
            rows.append(
                {
                    "example_id": item["example_id"],
                    "example_name": item["example_name"],
                    "category": item["category"],
                    "prompt": prompt,
                    "layer": layer_label,
                    "readout": readout,
                    "position": pos,
                    "source_token": source_tokens[pos],
                    "top1_token": tokenizer.decode(
                        [int(top1_ids[i])], clean_up_tokenization_spaces=False
                    ),
                    "rank": int(ranks[i]),
                }
            )

    for layer in lens.source_layers:
        rows_for(layer, "jacobian_lens", lens_logits[layer])
    rows_for(final_layer, "model_output", model_logits)

    del lens_logits, model_logits, mask, input_ids
    return rows


def combine_checkpoints() -> int:
    all_rows = []
    for ckpt in sorted(CKPT_DIR.glob("example_*.csv")):
        with open(ckpt, newline="") as f:
            reader = csv.DictReader(f)
            all_rows.extend(reader)
    if not all_rows:
        return 0
    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return len(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=len(SELECTED_NAMES))
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    pre = check_memory()
    log_memory("pre_run", pre)
    if not memory_is_safe(pre):
        print("Memory unsafe BEFORE model load; aborting without loading model.")
        sys.exit(2)

    prompts = load_selected_prompts()[: (1 if args.sanity_only else args.n_prompts)]

    pending = [p for p in prompts if not checkpoint_is_valid(checkpoint_path(p["example_id"]))]
    if not pending:
        print("All requested prompts already have valid checkpoints. Nothing to do.")
        combine_checkpoints()
        return

    print(f"Loading model {MODEL_NAME} on MPS ...", file=sys.stderr)
    import torch
    import transformers

    import jlens

    jlens.configure_logging()

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).to("mps")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    print(model, file=sys.stderr)

    print("Loading released pre-fitted lens ...", file=sys.stderr)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    print(lens, file=sys.stderr)
    assert lens.d_model == model.d_model, "lens/model d_model mismatch"

    post_load = check_memory()
    log_memory("post_model_load", post_load)
    if not memory_is_safe(post_load):
        print("Memory unsafe after model load; stopping before any prompts.")
        sys.exit(2)

    for item in pending:
        ckpt = checkpoint_path(item["example_id"])
        print(f"--- {item['example_id']} ({item['example_name']}) ---", file=sys.stderr)
        t0 = time.time()
        rows = process_prompt(model, tokenizer, lens, item)
        write_checkpoint_atomic(ckpt, rows)
        assert checkpoint_is_valid(ckpt), f"checkpoint write verification failed: {ckpt}"
        mark_completed(item["example_id"])
        dt = time.time() - t0
        print(f"    {len(rows)} rows written to {ckpt.name} in {dt:.1f}s", file=sys.stderr)

        del rows
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        snap = check_memory()
        log_memory(f"after_{item['example_id']}", snap)
        if not memory_is_safe(snap):
            print(
                f"Memory pressure threshold hit after {item['example_id']}; "
                "stopping gracefully before the next prompt."
            )
            break

    n_rows = combine_checkpoints()
    n_done = len(json.loads(COMPLETED_JSON.read_text())) if COMPLETED_JSON.exists() else 0
    print(f"Done. {n_done} prompt(s) completed; {n_rows} total rows in {RAW_CSV}.")


if __name__ == "__main__":
    main()
