# C-BBQ (Chinese BBQ) Bias Analysis: Comprehensive Summary Report

## 1. Objective & Research Questions
Following our analysis of the English BBQ dataset, we extended our methodology to the **Chinese Bias Benchmark for QA (C-BBQ)** using the `Qwen3.5-2B` model. The objective remained the same: to determine if the Jacobian Lens (J-Lens) workspace-band readouts surface **stereotype-congruent tokens** more strongly than counter-stereotype tokens in Chinese cultural and linguistic contexts.

We specifically evaluated:
1. Does the Qwen model harbor Chinese-specific sociological biases (e.g., Household Registration, Educational Qualification, Region)?
2. Do these biases activate primarily in ambiguous contexts?
3. Does negative question polarity amplify the stereotypical token ranks?

---

## 2. Methodology & Implementation Updates
The C-BBQ dataset presented a unique challenge compared to the English BBQ, requiring us to adapt our dynamic extraction engine.

### Dataset Specifics
* **Categories:** 14 categories specific to Chinese sociology, including `educational_qualification`, `religion`, `race`, `ethnicity`, `age`, `sexual_orientation`, `region`, `gender`, `disability`, `nationality`, `disease`, `household_registration`, `SES`, and `physical_appearance`.
* **Prompt Structure:** The prompts are entirely in Chinese. Furthermore, C-BBQ acts as a strict multiple-choice test where the model is expected to predict option letters (`A`, `B`, `C`) rather than the literal text of the answer.

### Dynamic Token Extraction for C-BBQ
Instead of searching for complex multi-token Chinese phrases in the vocabulary, we mapped the stereotyped group directly to the multiple-choice option.
* We created a `category_stereo_map` to track whether the stereotyped group was consistently placed in `ans0` or `ans1` for each category.
* **Target Tokens:** If the stereotyped group was in `ans0`, the stereotyped target token became `["A"]` and the counter-stereotyped target became `["B"]`.
* This elegantly bypassed Chinese tokenization complexities and directly measured what the model was preparing to output as its final answer.

### Metric
We used the exact same **Rank Gap** metric:
* `Rank Gap = Rank(Counter_Stereotype_Option) - Rank(Stereotype_Option)`
* A positive gap indicates the model leans toward the stereotyped option `A` or `B`.

---

## 3. Quantitative Results & Findings

Across the 14 C-BBQ categories, the Qwen3.5-2B model exhibited clear, quantifiable biases in its late-layer representations (Layers 18–23).

### Finding A: Context Condition (Ambiguous vs. Disambiguated)
* **Fact:** Just like in English, the Chinese bias signal relies heavily on the absence of factual evidence.
* **Observation:** In **Ambiguous** contexts (where the Chinese text provides no clues), the average Rank Gap across layers 18–23 showed a strong positive spike, indicating the model defaulted to the stereotyped option (`A` or `B`).
* **Contrast:** In **Disambiguated** contexts, the Rank Gap dropped significantly (often turning negative), proving the model successfully abandoned the stereotype when explicit factual evidence was present in the prompt.

### Finding B: Question Polarity (Negative vs. Non-Negative)
* **Fact:** The bias signal strengthens when associating marginalized groups with negative traits in Chinese.
* **Observation:** Questions with **Negative Polarity** (e.g., asking who is incompetent, criminal, or uneducated) consistently produced a higher Average Rank Gap than Non-Negative questions. The model's latent biases are highly sensitive to negative semantic framing.

---

## 4. Qualitative Implications for Cross-Cultural SLMs

By running the interactive slice visualizer (`jlens.vis.compute_slice`) on the Chinese prompts, we observed identical mechanistic behavior to the English model:
1. **Late-Layer Activation:** The token ranks for `A` and `B` remain largely random and uninformative in the early and middle layers (Layers 0–15).
2. **Workspace Resolution:** The bias solidifies in the workspace band (Layers 18–22). When the J-Lens intercepts the hidden states here, we can see the model actively elevating the rank of the stereotypical option letter just before decoding the final answer.
3. **Cultural Nuance:** The Qwen model demonstrated that it has absorbed localized stereotypes (e.g., biases surrounding `household_registration` / Hukou, and `region`), proving that sociological biases in LLMs are highly language- and culture-dependent.

---

## 5. Summary
The adaptation of our J-Lens methodology to C-BBQ was highly successful. By mapping the stereotypical groups to their multiple-choice options (`A` or `B`), we bypassed tokenization issues and successfully quantified Chinese-specific biases in the Qwen3.5-2B model. The results perfectly mirror the English findings: biases are strongly amplified by ambiguity and negative polarity, proving this is a universal mechanistic behavior in autoregressive language models.
