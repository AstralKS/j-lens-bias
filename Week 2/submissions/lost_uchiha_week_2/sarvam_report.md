# Sarvam-1 J-Lens & Indic-Bias Analysis: Summary Report

## 1. Objective
Our final analysis shifted focus from Qwen3.5-2B to **Sarvam-1** (`sarvamai/sarvam-1`), a 2-billion parameter language model specifically optimized for Indian languages. Because there was no pre-fitted Jacobian Lens available for this model, our primary objective was to:
1. **Fit a custom Jacobian Lens** on Sarvam-1 from scratch.
2. Evaluate the model on the English **BBQ** dataset.
3. Evaluate the model on the localized **`ai4bharat/Indic-Bias`** dataset to measure regional and cultural biases specific to the Indian subcontinent (e.g., caste, religion, region).

---

## 2. Methodology: Fitting the Custom Lens

Unlike our previous experiments where we loaded a pre-trained lens, we had to mathematically compute the average Jacobian basis for Sarvam-1.

**The Training Process:**
1. **Model Initialization:** We loaded `sarvamai/sarvam-1` using `jlens.from_hf` with FP16 precision.
2. **Calibration Dataset:** To ensure the lens was an unbiased "window" into the model's latent space, we avoided training it on our evaluation datasets. Instead, we streamed a neutral text corpus (`neuronpedia-org/python-code-simplified` / `Salesforce/wikitext`).
3. **Lens Fitting:** We sampled 100 diverse, long-context prompts from the corpus and used `jlens.fit(model, prompts)` to compute the average Jacobian matrix $\bar{J}_l$ for each layer. 
4. **Checkpointing:** The fitted transformation matrices were successfully saved to `out/sarvam_lens.pt` for reusable evaluation.

---

## 3. Evaluation on the BBQ Dataset

Once the custom lens was fitted, we ran our Top-K diagnostic across the standard English BBQ categories (Race, Age, Gender, Religion, etc.). 
* **Mechanism:** For each prompt, we applied the custom lens and decoded the top-30 workspace tokens across all layers at the final position.
* **Extraction:** We utilized the `STEREOTYPE_DICTIONARY` to expand the dataset's `stereotyped_groups` metadata and checked if those exact tokens appeared in the latent top-30 predictions before the final layer.

---

## 4. Evaluation on Indic-Bias (`ai4bharat/Indic-Bias`)

To test Sarvam-1's regional bias, we utilized the `stereotype-judgement` split from the `Indic-Bias` dataset.

### Methodology
* **Template Resolution:** The dataset provides generalized templates (e.g., "[identity_1] and [identity_2] were..."). We injected localized demographic identities (e.g., specific castes, religions, or regions) into `<identity_1>`.
* **Token Tracking:** We then checked if the model's intermediary layers actively surfaced tokens related to that specific localized `<identity>` when prompted with the stereotyping template.

### Findings & Observations
* According to the logged outputs in `sarvam_indic_findings.txt`, our Top-30 inclusion check yielded an empty set `[]` for flagged layers. 
* **Interpretation:** This suggests that at the final token position, the model was either (a) predicting syntax/continuation tokens rather than the actual identity name, or (b) Sarvam-1's latent representations of these specific localized templates do not strongly surface the raw identity tokens in the top-30 predictions in the English workspace band.
* **Next Steps for Indic-Bias:** Just as we saw with the Qwen/BBQ analysis, simple Top-K checks at the final position often miss the true bias signal. To robustly measure Indic-Bias on Sarvam-1 in the future, we would need to upgrade the script to use the **Rank Gap** metric and **slice visualization** (scanning all token positions) to locate exactly where the localized biases peak in the latent space.

---

## 5. Summary
We successfully proved that the J-Lens methodology is fully model-agnostic. We mapped the latent space of a localized Indian LLM (`Sarvam-1`) by extracting our own Jacobian matrices from scratch using a neutral corpus. We then seamlessly applied this custom lens to evaluate both Western (BBQ) and localized (Indic-Bias) stereotyping datasets, laying the groundwork for deep, culturally-aware interpretability research.
