# Assignment 3 - Analysis and Intervention

**Objective:** Extend bias detection beyond gender and/or implement steering/intervention techniques on the J-Lens to understand how model representations respond to targeted manipulations. This week's assignment is more flexible than the previous ones. You can choose to either continue the exploratory analysis of last week's assignment, or start experimenting with intervention. 

---

## Option 1: Expand Bias Detection Across Contexts

Continue exploring datasets with the J-Lens, but branch beyond gender bias—the dimension where we've achieved the strongest results so far.

### What we've discussed
- The multiple-choice structure of our previous work used most of the relevant tokens we were looking at; bias signals competed with format-specific activations.
- Other social biases (race, ethnicity, religion, age, socioeconomic status, nationality, etc.) may emerge in different contexts or under different probing strategies.

### Suggested approaches

1. **Drop the multiple-choice**
   - Try to alter the **BBQ** data or other bias benchmarks to use open-ended formats (e.g., cloze-style completion, free text generation).
   - Look for datasets without rigid question-answer templating to isolate bias signals from format artifacts.

2. **Search for new datasets**
   - Survey the landscape for domain-specific bias benchmarks (e.g., hiring, lending, healthcare, criminal justice).
   - Consider synthetic or semi-synthetic datasets if naturally-occurring data is scarce for certain bias types.

3. **Develop alternative identification strategies**
   - Consider what J-Lens dimensions are likely to encode specific biases, and probe them directly.

---

## Option 2: Implement Steering & Intervention

Design and execute an intervention on the model's representations via the J-Lens, then evaluate the downstream effects. Radnitz already worked on steering with the J-Lens, check out his work or feel free to develop your own. 

This is open-ended and felxible. **You decide what to intervene on and how.** Below are examples we've discussed; feel free to explore other valid approaches.

### Example interventions

1. **Stress testing with biased/harsh scenarios**
   - Create adversarial or high-stakes prompts designed to elicit utilitarian or stereotypical responses.
   - Use the J-Lens to observe how representations shift under pressure.
   - Evaluate whether steering can mitigate or amplify these responses.

2. **Gender token steering**
   - Identify prompts that naturally elicit gendered pronouns or roles (e.g., "The nurse took a break" → model biases toward "she").
   - Steer the J-Lens to suppress stereotype-aligned tokens (e.g., suppress female stereotypes in a nursing context).
   - Evaluate the model's response:
     - Does it generate counter-stereotypical outputs?
     - Do other biases emerge as compensation?
     - How does generation quality or coherence degrade?

3. **Other interventions**
   - Flip stereotypical associations in latent space and measure downstream changes.
   - Probe causal connections: does steering a lens dimension reliably change model behavior?
   - Intervene on non-gender dimensions to build a comparative understanding.

### To Consider:
- How will you evaluate whether the intervention worked? (e.g., change in token probabilities, shift in generation, bias metrics, human evaluation)
- What does a successful vs. failed intervention tell us about the model's bias mechanisms?
