# NLP-based Essay Scoring System for English Language Learners

Automatically scoring essays written by English Language Learners (grades 8–12) on six analytic writing dimensions, using the [Feedback Prize – English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) dataset.

This project combines three very different views of an essay — word and character patterns, 106 hand-built linguistic features, and a fine-tuned DeBERTa transformer — and stacks them. The focus throughout was **testing every assumption against cross-validation**, including the ones that turned out to be wrong.

## Results at a Glance

| Stage | MCRMSE |
|-------|--------|
| Baseline (predict the mean) | 0.6528 |
| TF-IDF + Ridge | 0.5288 |
| Classical stack (TF-IDF + linguistic features) | 0.5042 |
| DeBERTa-v3-base alone (768 tokens) | 0.4842 |
| **Final stack (all 6 models)** | **0.4716** |

*Metric: Mean Columnwise RMSE across the six targets, lower is better. A 27.8% error reduction over the baseline.*

## How I Framed the Problem

- **Six scores, one essay.** Each essay is rated 1.0–5.0 (in 0.5 steps) on cohesion, syntax, vocabulary, phraseology, grammar and conventions. The targets are strongly related (correlations 0.64–0.74), yet only **1.7%** of essays get the same score on all six, and **56%** have a 1-point spread. So they share a common "writing ability" signal but are genuinely distinct.
- **Scores cluster in the middle.** Most essays sit between 2.5 and 3.5, and scores of 1 or 5 are rare, which later shows up as the model's hardest cases.
- **Length is a clue, not the answer.** Word count correlates with scores (up to +0.27 for vocabulary) but barely with grammar (+0.08), so a model can't simply reward long essays.
- **Essays are long for transformers.** The median is 402 words and the 95th percentile is 788, so a 512-token model will cut off the end of many essays. This shaped a later experiment.
- **Only 3,911 essays**, so every modelling choice had to be validated carefully rather than assumed.

## My Thought Process

### 1. Get validation right before building anything
- **Wrote and tested my own MCRMSE function** with three sanity checks: a perfect prediction scores 0, predictions off by exactly 1 score 1.0, and predicting the mean equals the targets' standard deviation (0.6528).
- **Caught a stratification mistake.** With six targets, I used `MultilabelStratifiedKFold` to balance folds. Fed the raw scores, it made folds *more* unbalanced than plain random KFold (average imbalance 0.0846 vs 0.0329), because it treats values as binary labels. One-hot encoding the scores into 54 columns fixed it: **imbalance dropped to 0.0119**, about 3× better than random.
- **Folds were saved once and reused by every model**, including the DeBERTa notebook, which regenerated them and asserted an exact fold mean (3.1216) to prove they matched.

### 2. Build a strong, simple baseline and let the data correct me
TF-IDF + Ridge, improved one change at a time:

| Change | MCRMSE | What it taught me |
|--------|--------|-------------------|
| Words, default settings | 0.5721 | Starting point |
| Keep capitalisation | 0.5680 | Capitalisation errors are signal for learners, not noise |
| + word pairs (bigrams) | 0.5762 | Looked harmful; raising the feature cap didn't help (0.5768) |
| + character n-grams (3–5) | 0.5480 | Captures misspellings and word-form errors that whole-word features miss |
| Tune regularisation (alpha 10 → 3) | 0.5338 | Big gain from one parameter |
| **Retest bigrams with tuned alpha** | **0.5288** | Bigrams weren't bad; they needed a different alpha for the larger feature space |

That last row was the most useful lesson here: **a feature I'd rejected was actually good, and I only found out because I retested it after changing the regularisation.** Earlier conclusions depend on the settings they were tested under.

I also tried other learners on the same features: LinearSVR (0.5550), SVD + Ridge (0.5402) and SVD + LightGBM (0.5692) all lost to plain Ridge on the full sparse matrix. `min_df` barely mattered (0.5336–0.5354), so I didn't overtune it.

### 3. Engineer features that reflect how writing is actually judged
I parsed every essay once with spaCy and built **106 linguistic features** across 8 groups: descriptive, lexical diversity, syntactic, referential cohesion, readability, connectives, repetition and mechanics. What stood out:
- **Mechanics matter most.** The share of sentences starting with a lowercase letter was the single strongest feature (−0.41 with conventions).
- **Measure diversity correctly.** Raw type-token ratio showed almost no relationship with scores, because it naturally falls as essays get longer. Length-corrected versions (root TTR, MTLD) were among the strongest features (up to +0.40 with vocabulary).
- **Variety beats frequency.** Using *more types* of connectives correlated positively (+0.30 with cohesion), while simply using more connectives per sentence correlated negatively.
- **Repetition is a warning sign.** The share of an essay taken up by its 20 most frequent words correlated −0.39 with vocabulary.

### 4. Catch a confound in my own features
My "syntactic complexity" features (dependency distance, parse depth, clauses per sentence) looked strong, but they were mostly **measuring missing punctuation**: run-on sentences create artificially "complex" parses. Parse depth correlated **−0.82** with punctuation rate. Restricting to essays with normal sentence lengths cut dependency distance's correlation with syntax from −0.32 to −0.18, and clauses per sentence from −0.26 to −0.10.

So I rebuilt the group: per-sentence normalised measures, variety measures (unique dependency types, entropy), and the confound itself as an explicit feature (punctuation rate, run-on flag), so the model sees each signal for what it really is.

### 5. Value models for what they add, not how they score alone
All three feature-based models (Ridge, LightGBM, XGBoost) scored **~0.533, slightly worse than TF-IDF alone**. But blending them with TF-IDF 50/50 dropped the error to **0.5094**. The features capture different information from word patterns, which is exactly what an ensemble needs.

Different averaging methods (arithmetic, geometric, harmonic, quadratic) all landed within 0.001 of each other, so I didn't read anything into which "won". Switching from a weighted average to a **Ridge stacking model** gave a real gain (0.5095 → 0.5042), partly because each target's prediction could use the other targets' predictions (all 24 columns beat each target's own 4).

### 6. Fine-tune a transformer carefully
DeBERTa-v3-base with mean pooling and a regression head, trained on a single T4 GPU:
- **Head bias initialised to the target means**, so training starts from the baseline instead of random predictions.
- **Dynamic padding**, mixed precision and gradient checkpointing to fit long essays in 16 GB.
- **Evaluated every 40 steps.** Validation scores swung heavily within an epoch (e.g. 0.49 to 0.62 on one fold), so checking only at epoch ends would have missed the best points.
- **Guarded against silent bugs** with assertions that every validation row is filled exactly once and never overwritten.

Result: **0.4881** at 512 tokens, already better than the entire classical stack.

### 7. Test whether the truncation actually matters
Since many essays exceed 512 tokens, I retrained at **768 tokens** (201 min vs 122 min). It improved to 0.4842, and where the gain appeared made sense:

| Target | Gain from 768 tokens |
|--------|---------------------|
| Conventions | +0.0077 |
| Grammar | +0.0070 |
| Cohesion | +0.0056 |
| Vocabulary | −0.0005 |

Cohesion and conventions depend on the whole essay, while vocabulary quality is visible within the first few paragraphs.

### 8. Stack everything, and measure each piece's contribution
I compared 11 ways to combine the six models. Ridge (alpha 50) won at **0.4716**, with PLS and ElasticNet within 0.002, so I chose the simplest. Two informative results: a **simple average (0.4855) was worse than DeBERTa alone**, because weaker models need learned weights; and unpenalised linear regression was just as bad (0.4856), because the inputs are highly correlated.

An ablation showed where the gains came from:

| Stack | MCRMSE | Added |
|-------|--------|-------|
| Classical models only | 0.5042 | — |
| + DeBERTa 512 | 0.4732 | +0.0309 |
| + DeBERTa 768 | 0.4716 | +0.0016 |

The 768 model was 0.004 better on its own but added only 0.0016 to the stack: the two DeBERTa runs correlate at 0.97, so most of their signal overlaps.

## Error Analysis: Where the Model Still Struggles

| Target | Final RMSE | Improvement over baseline |
|--------|-----------|---------------------------|
| Cohesion | 0.4982 | 24.8% |
| Syntax | 0.4578 | 28.9% |
| Vocabulary | 0.4247 | 27.2% |
| Phraseology | 0.4704 | 28.3% |
| Grammar | 0.5108 | 27.0% |
| Conventions | 0.4679 | 30.3% |

- **Cohesion improved least**, which fits: judging how ideas connect across an essay is the most holistic skill.
- **The model plays it safe at the extremes.** Its predictions are 21% less spread out than the true scores. Essays scored below 2 are over-predicted by +0.44 on average, and essays scored 4.5+ are under-predicted by −0.76. This is expected from a squared-error loss on data concentrated around 3.
- **Long essays are harder** (error 0.395 vs 0.368 for essays over 450 words), with the biggest gap in phraseology, consistent with truncation still losing information.
- **Model disagreement doesn't predict error** (correlation 0.001), so disagreement between models can't be used as a confidence signal here.


## Tech Stack

Python · pandas · NumPy · scikit-learn · spaCy · textstat · lexicalrichness · LightGBM · XGBoost · PyTorch · Hugging Face Transformers (DeBERTa-v3-base) · iterative-stratification
