#!/usr/bin/env python3
"""Select a small, matched, audit-able set of official BBQ "Age" items for
the Week 2 J-Lens exploratory bias analysis.

Source of truth: the official nyu-mll/BBQ repository's data/Age.jsonl
(downloaded verbatim, not modified). This script never invents or edits BBQ
fields; it only groups rows that already share (question_index, ans0, ans1,
ans2) into "base items" (the same scenario across the 2x2 ambiguous/
disambiguated x negative/non-negative design) and derives the
stereotype/counter/unknown answer mapping from BBQ's own metadata:

  - answer_info[ansK] = [target_text, group_label]
  - additional_metadata.stereotyped_groups = list of group_label(s) that are
    "the" stereotyped group for this item
  - stereotype index: the ansK whose group_label is in stereotyped_groups
  - counter index:   the ansK whose group_label is not in stereotyped_groups
                      and not "unknown"
  - unknown index:   the ansK whose group_label is "unknown"

If any of the three roles is not uniquely determined for a base item, that
item is skipped (never guessed).

Writes:
  Week 2/data/age_selected_items.jsonl   - the raw BBQ rows actually used
                                            (verbatim fields + our derived
                                            role indices), one row per line
  Week 2/data/selection_manifest.json    - human-auditable summary: which
                                            base items were picked, why, and
                                            the exact metadata that justified
                                            each stereotype/counter mapping
"""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

WEEK2_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WEEK2_DIR / "data"
BBQ_AGE_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/Age.jsonl"
LOCAL_RAW_CACHE = DATA_DIR / "_raw_age_full.jsonl"  # not committed; cache only

CATEGORY = "Age"

# 10 diverse question templates (of the 25 available in BBQ/Age), chosen for
# lexical diversity of the stereotype/counter target words and to cover a
# spread of age-bias framings (memory, technology, work, risk-taking,
# attractiveness). Chosen BEFORE looking at any J-Lens output -- selection
# is keyed only on question_index identity, not on any model result.
CHOSEN_QUESTION_INDICES = [
    "1",  # grandfather / grandson - technology
    "2",  # grandfather / grandson - mobility
    "3",  # elderly / younger - memory
    "4",  # grandpa / son - memory
    "5",  # Millennial / Boomer - work engagement
    "9",  # grandfather / grandson - values
    "13",  # son / elderly mother - recklessness
    "16",  # teen / middle-aged guy - substance use
    "19",  # middle-aged guy / teen - "out of touch"
    "22",  # 15 year old / 40-year-old - recklessness
]


def fetch_age_jsonl() -> list[dict]:
    if not LOCAL_RAW_CACHE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official BBQ Age.jsonl from {BBQ_AGE_URL} ...")
        with urllib.request.urlopen(BBQ_AGE_URL, timeout=60) as resp:
            body = resp.read()
        LOCAL_RAW_CACHE.write_bytes(body)
    return [json.loads(line) for line in LOCAL_RAW_CACHE.read_text().splitlines() if line]


def derive_roles(rows: list[dict]) -> dict | None:
    """Given the 4 matched rows of one base item (all share ans0/ans1/ans2 and
    answer_info), derive stereotype/counter/unknown ansK indices. Returns None
    (skip) if metadata does not uniquely determine all three roles."""
    r0 = rows[0]
    ai = r0["answer_info"]
    stereotyped_groups = set(r0["additional_metadata"]["stereotyped_groups"])
    labels = {k: ai[k][1] for k in ("ans0", "ans1", "ans2")}

    unk = [k for k, v in labels.items() if v.lower() == "unknown"]
    stereo = [k for k, v in labels.items() if v in stereotyped_groups]
    counter = [
        k for k, v in labels.items() if v not in stereotyped_groups and v.lower() != "unknown"
    ]
    if len(unk) != 1 or len(stereo) != 1 or len(counter) != 1:
        return None
    return {
        "unknown_key": unk[0],
        "stereotype_key": stereo[0],
        "counter_key": counter[0],
        "stereotype_group_label": labels[stereo[0]],
        "counter_group_label": labels[counter[0]],
        "stereotype_target_text": ai[stereo[0]][0],
        "counter_target_text": ai[counter[0]][0],
        "stereotyped_groups_metadata": sorted(stereotyped_groups),
    }


def main() -> None:
    rows = fetch_age_jsonl()
    print(f"Loaded {len(rows)} raw {CATEGORY} rows.")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["question_index"], r["ans0"], r["ans1"], r["ans2"])
        groups[key].append(r)

    # One base item per chosen question_index: the first clean (size-4) group
    # in file order, so selection is deterministic and reproducible.
    picked_by_qi: dict[str, list[dict]] = {}
    for key, group_rows in groups.items():
        qi = key[0]
        if qi not in CHOSEN_QUESTION_INDICES or qi in picked_by_qi:
            continue
        if len(group_rows) != 4:
            continue
        conditions = {(r["context_condition"], r["question_polarity"]) for r in group_rows}
        if conditions != {
            ("ambig", "neg"),
            ("ambig", "nonneg"),
            ("disambig", "neg"),
            ("disambig", "nonneg"),
        }:
            continue
        roles = derive_roles(group_rows)
        if roles is None:
            print(f"SKIP question_index={qi}: ambiguous stereotype/counter/unknown mapping")
            continue
        picked_by_qi[qi] = sorted(group_rows, key=lambda r: (r["context_condition"], r["question_polarity"]))

    missing = [qi for qi in CHOSEN_QUESTION_INDICES if qi not in picked_by_qi]
    if missing:
        print(f"WARNING: no clean base item found for question_index {missing}")

    selected_rows: list[dict] = []
    manifest_items: list[dict] = []
    for order, qi in enumerate([q for q in CHOSEN_QUESTION_INDICES if q in picked_by_qi], start=1):
        group_rows = picked_by_qi[qi]
        roles = derive_roles(group_rows)
        base_item_id = f"age_{order:02d}_qi{qi}"
        for r in group_rows:
            out_row = dict(r)  # verbatim official fields
            out_row["base_item_id"] = base_item_id
            out_row.update({f"role_{k}": v for k, v in roles.items()})
            selected_rows.append(out_row)
        manifest_items.append(
            {
                "base_item_id": base_item_id,
                "question_index": qi,
                "example_ids": [r["example_id"] for r in group_rows],
                "question": group_rows[0]["question"],
                "stereotyped_groups_metadata": roles["stereotyped_groups_metadata"],
                "stereotype_answer": {
                    "key": roles["stereotype_key"],
                    "target_text": roles["stereotype_target_text"],
                    "group_label": roles["stereotype_group_label"],
                },
                "counter_answer": {
                    "key": roles["counter_key"],
                    "target_text": roles["counter_target_text"],
                    "group_label": roles["counter_group_label"],
                },
                "unknown_answer_key": roles["unknown_key"],
                "justification": (
                    f"answer_info label '{roles['stereotype_group_label']}' appears in "
                    f"additional_metadata.stereotyped_groups={roles['stereotyped_groups_metadata']} "
                    f"-> stereotype={roles['stereotype_key']}; label "
                    f"'{roles['counter_group_label']}' does not and is not 'unknown' -> "
                    f"counter={roles['counter_key']}; remaining ansK labeled 'unknown' -> "
                    f"unknown={roles['unknown_key']}."
                ),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = DATA_DIR / "age_selected_items.jsonl"
    with open(out_jsonl, "w") as f:
        for row in selected_rows:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "category": CATEGORY,
        "source": BBQ_AGE_URL,
        "n_base_items": len(manifest_items),
        "n_rows": len(selected_rows),
        "mapping_rule": (
            "stereotype/counter/unknown roles derived solely from official BBQ "
            "answer_info[ansK] = [target_text, group_label] cross-checked against "
            "additional_metadata.stereotyped_groups; items skipped if not unique."
        ),
        "category_choice_rationale": (
            "Age was chosen as the primary category because every one of its 3680 "
            "rows has an unambiguous 1:1:1 stereotype/counter/unknown mapping "
            "(verified programmatically: 0/3680 problem rows), it offers 25 distinct "
            "question templates with short, single/double-word target strings that "
            "tokenize cleanly, and it natively supports the full "
            "ambiguous/disambiguated x negative/non-negative 2x2 design."
        ),
        "items": manifest_items,
    }
    (DATA_DIR / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(selected_rows)} rows ({len(manifest_items)} base items) to {out_jsonl}")
    print(f"Wrote manifest to {DATA_DIR / 'selection_manifest.json'}")


if __name__ == "__main__":
    main()
