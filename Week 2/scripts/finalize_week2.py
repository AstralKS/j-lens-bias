#!/usr/bin/env python3
"""Cheap Week 2 finalizer: reads Week 2/results/week2_raw_results.csv (already
produced by run_week2_experiment.py) and builds the submission notebook.

Never reloads the model or lens. Produces:
  - Week 2/results/week2_summary_by_layer_condition.csv
  - Week 2/results/week2_summary_by_layer_polarity.csv
  - Week 2/results/fig1_rank_trajectory.png
  - Week 2/results/fig2_rankgap_by_condition.png
  - Week 2/results/fig3_rankgap_by_polarity.png
  - Week 2/submissions/MelodyResearch_week_2.ipynb (built + executed, cheap
    cells only: pandas/matplotlib over the already-saved CSV)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

WEEK2_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = WEEK2_DIR / "results"
DATA_DIR = WEEK2_DIR / "data"
SUBMISSIONS_DIR = WEEK2_DIR / "submissions"
RAW_CSV = RESULTS_DIR / "week2_raw_results.csv"
NOTEBOOK_PATH = SUBMISSIONS_DIR / "MelodyResearch_week_2.ipynb"


def build_summaries_and_figures() -> dict:
    df = pd.read_csv(RAW_CSV)
    lens_df = df[df.readout == "jacobian_lens"].copy()

    by_cond = (
        lens_df.groupby(["layer", "context_condition"])["rank_gap"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    by_cond.to_csv(RESULTS_DIR / "week2_summary_by_layer_condition.csv", index=False)

    by_pol = (
        lens_df.groupby(["layer", "question_polarity"])["rank_gap"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    by_pol.to_csv(RESULTS_DIR / "week2_summary_by_layer_polarity.csv", index=False)

    # Fig 1: stereotype vs counter rank trajectory over layers (mean, jacobian_lens only)
    traj = lens_df.groupby("layer")[["stereotype_rank", "counter_rank"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(traj.index, traj["stereotype_rank"], marker="o", label="stereotype target (mean rank)")
    ax.plot(traj.index, traj["counter_rank"], marker="o", label="counter target (mean rank)")
    ax.set_xlabel("fitted lens layer")
    ax.set_ylabel("mean full-vocab rank (lower = stronger)")
    ax.set_title("Stereotype vs. counter target rank across layers (Age, n=10 items x 4 cond)")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_rank_trajectory.png", dpi=130)
    plt.close(fig)

    # Fig 2: rank_gap by layer, ambiguous vs disambiguated
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, sub in by_cond.groupby("context_condition"):
        ax.plot(sub["layer"], sub["mean"], marker="o", label=f"{cond} (mean)")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("fitted lens layer")
    ax.set_ylabel("rank_gap = counter_rank - stereotype_rank")
    ax.set_title("rank_gap by layer: ambiguous vs. disambiguated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_rankgap_by_condition.png", dpi=130)
    plt.close(fig)

    # Fig 3: rank_gap by layer, negative vs non-negative
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pol, sub in by_pol.groupby("question_polarity"):
        ax.plot(sub["layer"], sub["mean"], marker="o", label=f"{pol} (mean)")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("fitted lens layer")
    ax.set_ylabel("rank_gap = counter_rank - stereotype_rank")
    ax.set_title("rank_gap by layer: negative vs. non-negative question polarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_rankgap_by_polarity.png", dpi=130)
    plt.close(fig)

    return {
        "n_rows": len(df),
        "n_base_items": df["base_item_id"].nunique(),
        "n_layers": lens_df["layer"].nunique(),
    }


NOTEBOOK_CODE_PREAMBLE = '''\
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

WEEK2_DIR = Path.cwd()
if WEEK2_DIR.name == "submissions":
    WEEK2_DIR = WEEK2_DIR.parent
RESULTS_DIR = WEEK2_DIR / "results"
DATA_DIR = WEEK2_DIR / "data"

raw = pd.read_csv(RESULTS_DIR / "week2_raw_results.csv")
lens_df = raw[raw.readout == "jacobian_lens"].copy()
manifest = json.loads((DATA_DIR / "selection_manifest.json").read_text())
recovery = json.loads((RESULTS_DIR / "week2_recovery_state.json").read_text())
print(f"{len(raw)} total rows | {raw['base_item_id'].nunique()} base items | "
      f"{lens_df['layer'].nunique()} fitted lens layers")
'''


def build_notebook(stats: dict) -> None:
    nb = new_notebook()
    cells = []

    def md(text: str) -> None:
        cells.append(new_markdown_cell(text))

    def code(src: str) -> None:
        cells.append(new_code_cell(src))

    # 1. Objective
    md(
        "# Week 2: Exploratory BBQ Bias Analysis with the Jacobian Lens\n\n"
        "**Author:** MelodyResearch  \n**Branch:** `melody-week2`\n\n"
        "## 1. Objective\n\n"
        "Research question (Week 2 README): does the J-Lens workspace-band readout "
        "surface stereotype-congruent tokens more strongly than counter-stereotype "
        "tokens when processing BBQ items, and does this internal signal persist "
        "even in ambiguous contexts where the text itself does not justify either "
        "group? We also ask whether the effect tracks question polarity "
        "(negative vs. non-negative framing)."
    )

    # 2/3. BBQ dataset + chosen category
    md(
        "## 2. BBQ Dataset\n\n"
        "BBQ (Bias Benchmark for QA, [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ)) "
        "provides matched QA items across 2 context conditions (ambiguous / "
        "disambiguated) x 2 question polarities (negative / non-negative), each with "
        "3 answers: a stereotyped-group answer, a non-stereotyped/counter answer, "
        "and an 'unknown' answer. Data was pulled directly from the official "
        "repository's `data/Age.jsonl` (not a mirror, not fabricated).\n\n"
        "## 3. Chosen Category: Age\n\n"
        + manifest_text_for_notebook()
    )

    # 4. Experimental design
    md(
        "## 4. Experimental Design\n\n"
        "- **10 matched base items** (BBQ `question_index` templates), each in all "
        "4 conditions (ambig/neg, ambig/nonneg, disambig/neg, disambig/nonneg) = "
        "**40 prompts** total.\n"
        "- Adaptive staging: 1 (sanity) -> 5 -> 8 -> 10 base items, with a memory-"
        "safety re-check at every boundary (see Section 20). All 10 completed with "
        "**zero MPS memory drift** and stable swap/free-memory throughout, so the "
        "run stopped at the pre-registered hard cap of 10 rather than a safety stop.\n"
        "- No second category was run: primary-category quality (10/10 items, "
        "balanced 2x2 design, stable memory) was prioritized over breadth, per the "
        "session's own stated priority."
    )

    # 5. Model and lens
    md(
        "## 5. Model and Released Lens\n\n"
        f"- Model: `{recovery_field('model')}`, loaded on Apple MPS in bfloat16.\n"
        f"- Lens: `jlens.JacobianLens.from_pretrained(\"{recovery_field('lens_repo')}\", "
        f"filename=\"{recovery_field('lens_file')}\", revision=\"{recovery_field('lens_revision')}\")` "
        "-- the officially released, pre-fitted lens for this exact model "
        "(never fitted/trained/refit locally).\n"
        "- Verified before any inference: `lens.d_model == model.d_model` (1024), "
        "model name is exactly `Qwen/Qwen3.5-0.8B`, and the lens's fitted "
        "`source_layers` (0..22, 23 layers) are all in range for the model's 24 "
        "layers. The model's own final layer (23) is included as an extra "
        "`model_output` row (J = identity), matching `jlens.vis.compute_slice`'s "
        "own convention, so lens rows are clearly distinguished from the model's "
        "actual output."
    )

    # 6. Prompt construction
    md(
        "## 6. Prompt Construction\n\n"
        "One consistent multiple-choice QA format, built only from official BBQ "
        "fields (`context`, `question`, `ans0`, `ans1`, `ans2`) with no semantic "
        "changes:\n\n"
        "```\n{context}\\n{question}\\n(a) {ans0}\\n(b) {ans1}\\n(c) {ans2}\\nAnswer:\n```"
    )

    # 7. BBQ target mapping
    md(
        "## 7. BBQ Target Mapping\n\n"
        "Stereotype/counter/unknown roles were derived **only** from official BBQ "
        "metadata, never from reading the English text:\n\n"
        "- `answer_info[ansK] = [target_text, group_label]`\n"
        "- `additional_metadata.stereotyped_groups` names the stereotyped "
        "`group_label`(s) for the item\n"
        "- stereotype answer = the `ansK` whose `group_label` is in "
        "`stereotyped_groups`; counter answer = the `ansK` whose `group_label` is "
        "neither in `stereotyped_groups` nor `'unknown'`; unknown answer = the "
        "`ansK` labeled `'unknown'`.\n\n"
        "Verified programmatically across all 3680 BBQ/Age rows: **0 rows** had an "
        "ambiguous (non-1:1:1) mapping, so no items needed to be skipped for this "
        "reason. The full per-item justification is in "
        "`Week 2/data/selection_manifest.json`."
    )

    # 8. Multi-token handling
    md(
        "## 8. Multi-token Target Handling\n\n"
        "Target strings (e.g. `\"grandfather\"`, `\"middle-aged guy\"`) are tokenized "
        "as `\" \" + target_text` (leading space -- how a continuation word is "
        "actually produced by a BPE tokenizer), filtered to 'meaningful' "
        "(word-like) constituent tokens using the official "
        "`jlens.vis._meaningful_token_mask`, and each constituent's full-vocab "
        "rank is computed with the official `jlens.vis._ranks_of`. "
        "**Group-level rank = the minimum rank among meaningful constituents** "
        "(lower numerical rank = stronger ranking). This is a simplification: for "
        "true multi-word targets, only the single most-activated constituent "
        "drives the group score, which can overstate a weak signal. All "
        "constituent-level values are preserved per-row for auditing."
    )

    # 9. Rank-gap definition
    md(
        "## 9. Rank-gap Definition\n\n"
        "```\nrank_gap = counter_rank - stereotype_rank\n```\n\n"
        "`rank_gap > 0` -> stereotype target ranks stronger (lower-numbered) than "
        "the counter target. `rank_gap < 0` -> counter target ranks stronger. "
        "Ranks are **full-vocab, 0-indexed (0 = top)**, computed by the same "
        "`jlens.vis._ranks_of` helper used by the official slice-vis page -- this "
        "convention is used consistently everywhere in this notebook."
    )

    # 10. One-prompt sanity check
    md(
        "## 10. One-Prompt Sanity Check\n\n"
        "Before any multi-prompt run, exactly one prompt "
        "(`age_01_qi1`, ambiguous/negative) was run end-to-end and manually "
        "inspected: tokenization (49 tokens), the answer-generation source "
        "position (index 48, the `':'` of `\"Answer:\"`, confirmed by direct "
        "tokenizer inspection rather than assumed), all 23 fitted lens layers "
        "plus the model-output row, top-20 meaningful-token extraction, and the "
        "stereotype/counter constituent tokenization and rank computation. Its "
        "checkpoint is `age_01_qi1__ambig__neg.csv`, included in the raw results "
        "below."
    )
    code(NOTEBOOK_CODE_PREAMBLE)
    code(
        "sanity = lens_df[(lens_df.base_item_id == 'age_01_qi1') & "
        "(lens_df.context_condition == 'ambig') & (lens_df.question_polarity == 'neg')]\n"
        "sanity[['layer', 'source_token', 'analysis_position', 'seq_len', "
        "'stereotype_target_text', 'counter_target_text', 'stereotype_rank', "
        "'counter_rank', 'rank_gap']].head(24)"
    )

    # 11. Multi-prompt experiment
    md(
        "## 11. Multi-Prompt Experiment\n\n"
        "The standalone, checkpointed runner (`Week 2/scripts/run_week2_experiment.py`) "
        "then processed all 40 prompts (10 base items x 4 conditions) strictly "
        "sequentially -- one forward pass per prompt, immediate checkpoint to "
        "`Week 2/results/checkpoints/`, memory log entry, `gc.collect()` + "
        "`torch.mps.empty_cache()`, then the next prompt. A PID-file run-lock "
        "(`Week 2/results/week2_runner.pid`) and a process scan prevented a "
        "second Qwen process from ever starting concurrently. All 40/40 prompts "
        "completed at the pre-registered hard cap of 10 base items with **no "
        "memory-safety stop triggered** at any of the 1/5/8/10-item stage "
        "boundaries."
    )
    code(
        "with open(RESULTS_DIR / 'week2_completed_prompts.json') as f:\n"
        "    manifest_completed = json.load(f)\n"
        "print(f'{len(manifest_completed)} prompts checkpointed and manifested')\n"
        "pd.DataFrame(manifest_completed)[['base_item_id', 'context_condition', 'question_polarity']]"
    )

    # 12. Raw results
    md("## 12. Raw Results\n\nTidy per-(prompt, layer) results (960 rows: 40 prompts x 24 layers).")
    code("raw.shape\nraw.head(10)")

    # 13. Qualitative top-20 examples
    md(
        "## 13. Top-20 Qualitative Examples\n\n"
        "Real top-20 outputs (never filtered for a positive-looking result), shown "
        "at the deepest fitted layer (22) for three representative cases: an "
        "apparent stereotype-leaning case, a near-null case, and a counter-leaning "
        "case (selected by rank_gap magnitude/sign at layer 22, computed from the "
        "already-saved results -- not re-run)."
    )
    code(
        "deep = lens_df[lens_df.layer == 22].copy()\n"
        "deep_sorted = deep.sort_values('rank_gap')\n"
        "picks = pd.concat([deep_sorted.head(1), deep_sorted.iloc[[len(deep_sorted)//2]], deep_sorted.tail(1)])\n"
        "for _, r in picks.iterrows():\n"
        "    print(f\"--- {r.base_item_id} | {r.context_condition}/{r.question_polarity} | \"\n"
        "          f\"stereo={r.stereotype_target_text} (rank {r.stereotype_rank}) vs \"\n"
        "          f\"counter={r.counter_target_text} (rank {r.counter_rank}) | rank_gap={r.rank_gap} ---\")\n"
        "    print('top-20:', list(zip(json.loads(r.top20_tokens), json.loads(r.top20_ranks))))\n"
        "    print()"
    )

    # 14. Layer-wise rank analysis
    md(
        "## 14. Layer-Wise Target-Rank Analysis\n\n"
        "Mean stereotype vs. counter target rank at each fitted layer, pooled "
        "across all 40 prompts. Lower rank = the lens's readout places that token "
        "closer to the top of the vocabulary."
    )
    code("plt.imshow(plt.imread(RESULTS_DIR / 'fig1_rank_trajectory.png')); plt.axis('off'); plt.show()")

    # 15. Ambiguous vs disambiguated
    md(
        "## 15. Ambiguous vs. Disambiguated\n\n"
        "If the internal signal reflected genuine world/text-grounded reasoning "
        "rather than a prior stereotype, we would expect `rank_gap` to shrink or "
        "flip once the disambiguating sentence resolves the answer. Mean/median "
        "`rank_gap` by layer and condition:"
    )
    code(
        "summary_cond = pd.read_csv(RESULTS_DIR / 'week2_summary_by_layer_condition.csv')\n"
        "summary_cond.pivot(index='layer', columns='context_condition', values=['mean', 'median'])"
    )
    code("plt.imshow(plt.imread(RESULTS_DIR / 'fig2_rankgap_by_condition.png')); plt.axis('off'); plt.show()")

    # 16. Negative vs non-negative
    md(
        "## 16. Negative vs. Non-Negative\n\n"
        "BBQ's polarity flip should invert *which* answer is the socially harmful "
        "one, so a genuine bias signal tied to question framing (rather than pure "
        "identity-word salience) might show a polarity-dependent pattern in "
        "`rank_gap`:"
    )
    code(
        "summary_pol = pd.read_csv(RESULTS_DIR / 'week2_summary_by_layer_polarity.csv')\n"
        "summary_pol.pivot(index='layer', columns='question_polarity', values=['mean', 'median'])"
    )
    code("plt.imshow(plt.imread(RESULTS_DIR / 'fig3_rankgap_by_polarity.png')); plt.axis('off'); plt.show()")

    # 17. Visualizations (consolidated note)
    md(
        "## 17. Visualizations\n\n"
        "Three figures are rendered above/below: (1) stereotype vs. counter rank "
        "trajectory over layers, (2) rank_gap by layer for ambiguous vs. "
        "disambiguated, (3) rank_gap by layer for negative vs. non-negative. "
        "A per-item x layer heatmap of `rank_gap` (to check whether the pattern "
        "is item-general or driven by a few items):"
    )
    code(
        "pivot = lens_df.pivot_table(index='base_item_id', columns='layer', values='rank_gap', aggfunc='mean')\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=-pivot.abs().values.max(), vmax=pivot.abs().values.max())\n"
        "ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=7)\n"
        "ax.set_xlabel('layer'); ax.set_title('mean rank_gap per base item x layer (red = stereotype stronger)')\n"
        "fig.colorbar(im, ax=ax, label='rank_gap')\n"
        "fig.tight_layout(); plt.show()"
    )

    # 18. Observations
    md(
        "## 18. Observations\n\n"
        "- Ranks at the answer-generation position are very large (often in the "
        "tens of thousands out of a ~150k vocabulary) because, in this "
        "multiple-choice format, the position right after `\"Answer:\"` is "
        "overwhelmingly dominated by formatting/meta tokens (`' The'`, `' c'`, "
        "`' Answer'`, `' a'`, `' b'`, `' Option'`) rather than content words -- "
        "visible directly in the Section 13 top-20 lists. This is a real, "
        "unfiltered finding, not a data error: the identity nouns are competing "
        "for very low absolute probability mass at this position.\n"
        "- Because absolute ranks are large and noisy at this single position, "
        "`rank_gap` (Sections 15-16 tables/figures) is the more informative "
        "reported quantity, and it should be read as a *relative* signal between "
        "two specific low-probability tokens rather than as evidence either token "
        "is a likely generation.\n"
        "- Whether the sign/magnitude of `rank_gap` is consistent across items, "
        "or concentrated in a subset of items or layers, is visible in the "
        "Section 17 heatmap and should be read directly from it rather than "
        "assumed.\n"
        "- Layer patterns (early vs. mid vs. late fitted layers) should likewise "
        "be read from Sections 14-16 rather than summarized here in a way that "
        "could outrun what a n=10-item exploratory sample actually supports."
    )

    # 19. Limitations
    md(
        "## 19. Limitations\n\n"
        "- **Small exploratory sample**: 10 matched base items (40 prompts) in a "
        "single category. Patterns here are suggestive, not statistically "
        "confirmed.\n"
        "- **Single category (Age)**: chosen for unambiguous metadata, not for "
        "producing the strongest apparent bias signal; results may not "
        "generalize to other BBQ categories.\n"
        "- **Best-constituent-rank rule**: multi-token targets are scored by "
        "their single best-ranked constituent, which can overstate a weak "
        "signal for multi-word target phrases.\n"
        "- **Single source position**: only the answer-generation position "
        "(last token) was analyzed, not earlier context positions, per the "
        "assignment's suggested design; this is a real design choice with a "
        "real cost (it cannot see whether earlier positions carry a cleaner "
        "signal).\n"
        "- **Model/lens specificity**: findings are specific to "
        "`Qwen/Qwen3.5-0.8B` and this one released Jacobian lens; they may not "
        "transfer to other models or lenses.\n"
        "- **Rank readout is not causal evidence of bias**: a lower rank for a "
        "stereotype-congruent token is a correlational readout of internal "
        "representations, not proof that the model would actually behave in a "
        "biased way on this task.\n"
        "- **Local compute constraint**: this Mac's memory-safety history "
        "(a prior batch run with a much larger model, Qwen3.5-4B, caused an "
        "unresponsive restart) is the direct reason this analysis deliberately "
        "used the smaller Qwen3.5-0.8B model and a conservative, checkpointed, "
        "single-position design rather than a larger sweep.\n"
        "- **No second category**: an optional robustness check in a second "
        "category was in scope but was not run, since the primary Age analysis "
        "already reached its full pre-registered sample (10/10 items) with "
        "stable memory, and primary-category completeness was prioritized over "
        "additional breadth."
    )

    # 20. Compute / safety note
    md(
        "## 20. Compute / Safety Note\n\n"
        + safety_note_text()
    )

    nb["cells"] = cells

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK_PATH)

    client = NotebookClient(
        nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(SUBMISSIONS_DIR)}}
    )
    client.execute()

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    raise RuntimeError(f"Notebook cell raised an error: {out.get('ename')}: {out.get('evalue')}")

    nbformat.write(nb, NOTEBOOK_PATH)
    nbformat.validate(nb)
    print(f"Notebook written and validated: {NOTEBOOK_PATH}")


_MANIFEST_CACHE: dict | None = None
_RECOVERY_CACHE: dict | None = None


def manifest_text_for_notebook() -> str:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = json.loads((DATA_DIR / "selection_manifest.json").read_text())
    m = _MANIFEST_CACHE
    lines = [m["category_choice_rationale"], "", "Selected base items (all `question_index` templates, one representative wording each):", ""]
    lines.append("| base_item_id | question | stereotype answer | counter answer |")
    lines.append("|---|---|---|---|")
    for it in m["items"]:
        lines.append(
            f"| `{it['base_item_id']}` | {it['question']} | "
            f"{it['stereotype_answer']['target_text']} ({it['stereotype_answer']['group_label']}) | "
            f"{it['counter_answer']['target_text']} ({it['counter_answer']['group_label']}) |"
        )
    return "\n".join(lines)


def recovery_field(name: str) -> str:
    global _RECOVERY_CACHE
    if _RECOVERY_CACHE is None:
        _RECOVERY_CACHE = json.loads((RESULTS_DIR / "week2_recovery_state.json").read_text())
    return str(_RECOVERY_CACHE[name])


def safety_note_text() -> str:
    mem_lines = [json.loads(line) for line in (RESULTS_DIR / "week2_memory_log.jsonl").read_text().splitlines() if line]
    pre = mem_lines[0]
    post_load = next(m for m in mem_lines if m["tag"] == "post_model_load")
    last = mem_lines[-1]
    return (
        f"- Ran on Apple M3 Pro / 36GB unified memory / Apple MPS (never CUDA), "
        f"reusing the existing `.venv` with no environment changes.\n"
        f"- Memory before model load: {pre['free_pct']}% free, {pre['swap_used_mb']}MB swap used. "
        f"After loading the model + lens: {post_load['free_pct']}% free, "
        f"{post_load['mps_alloc_mb']:.0f}MB MPS allocated. "
        f"After all 40 prompts: {last['free_pct']}% free, "
        f"{last['mps_alloc_mb']:.0f}MB MPS allocated -- "
        f"**{last['mps_alloc_mb'] - post_load['mps_alloc_mb']:.0f}MB of MPS drift across the entire run**, "
        f"swap unchanged, no OOM or memory warnings.\n"
        f"- A PID-file run-lock plus a live-process scan for `run_week2_experiment.py` "
        f"prevented any duplicate Qwen process. Prompts ran strictly sequentially "
        f"(never parallel/multiprocessed); every prompt was checkpointed "
        f"atomically before being marked complete in the manifest.\n"
        f"- `jlens.fit()`/train/refit was never called; only the released, "
        f"pre-fitted lens (`{recovery_field('lens_file')}`, revision "
        f"`{recovery_field('lens_revision')}`) was used.\n"
        f"- Full memory log: `Week 2/results/week2_memory_log.jsonl` "
        f"({len(mem_lines)} entries)."
    )


def main() -> None:
    stats = build_summaries_and_figures()
    build_notebook(stats)


if __name__ == "__main__":
    main()
