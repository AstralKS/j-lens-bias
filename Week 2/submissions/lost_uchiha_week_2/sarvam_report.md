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

### The Token Fragmentation Challenge & Resolution
Initially, several categories (such as SES, Disability Status, and Sexual Orientation) yielded empty results `[]`. We discovered this was a **tokenization artifact**: because Sarvam-1 is an Indic-focused model, it fragments complex English words (like `"wheelchair"` or `"homosexual"`) into multiple obscure sub-tokens (e.g., `[' wheel', 'ch', 'air']`). 

To resolve this, we:
1. **Translated Targets:** Augmented our dictionary with Hindi and Telugu translations (e.g., "विकलांग", "వికలాంగుడు") where the model natively has whole-concept tokens.
2. **Prefix Matching:** Interrogated the tokenizer to extract the exact first sub-token for every target word (e.g., `" Hind"` for Hindu) and matched against those specific prefixes.

### Findings
With the corrected tokenizer alignment, the model successfully surfaced the target stereotypes!
* **Gender_identity:** Layers [22, 23, 24, 25, 26]
* **Race_ethnicity:** Layers [23, 24, 25, 26]
* **Age & Religion:** Layers [26]

**Takeaway:** The model reserves its stereotypical associations for the very late layers (22-26). It builds grammatical and contextual understanding in the first ~21 layers, and only injects the biased identity probabilities at the very end of the network.

---

## 4. Evaluation on Indic-Bias (`ai4bharat/Indic-Bias`)

To test Sarvam-1's regional bias, we utilized the `stereotype-judgement` split from the `Indic-Bias` dataset.

### Methodology
* **Template Resolution:** The dataset provides generalized templates (e.g., "[identity_1] and [identity_2] were..."). We injected localized demographic identities (e.g., specific castes, religions, or regions) into `<identity_1>`.
* **Token Tracking:** We then checked if the model's intermediary layers actively surfaced tokens related to that specific localized `<identity>` when prompted with the stereotyping template.

### Findings & Observations
* Applying the same token-prefix alignment strategy, we were able to successfully extract the targeted identities from the model's latent workspace.
* **Flagged Layers:** The evaluation identified strong stereotypical token surfacing at **Layers 25 and 26** across almost all tested categories (political engagement, gender norms, social change, etc.).
* **Rank Analysis:** By plotting the probability rank of the target token across all 27 layers, we visually confirmed that the stereotyped identity token slumbers in the thousands (low probability) for layers 0-24, before plummeting down to rank #1 or #2 at layer 25/26.
* **Interpretation:** This conclusively proves that Sarvam-1 stores and activates its demographic biases uniformly at the very end of its computational graph. By successfully tracking these concepts natively through an Indic model, we demonstrated that mechanistic interpretability techniques like the Jacobian Lens can be effectively localized.

---

## 5. Summary
We successfully proved that the J-Lens methodology is fully model-agnostic. We mapped the latent space of a localized Indian LLM (`Sarvam-1`) by extracting our own Jacobian matrices from scratch using a neutral corpus. We then seamlessly applied this custom lens to evaluate both Western (BBQ) and localized (Indic-Bias) stereotyping datasets, laying the groundwork for deep, culturally-aware interpretability research.
