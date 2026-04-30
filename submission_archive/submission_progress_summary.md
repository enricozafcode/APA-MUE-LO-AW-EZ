# Submission Progress Summary (Submissions 1–6, plus v1.6 / v1.7)

## Submission 1 (`sub_1`)

This was mainly a test run.  
The idea was to generate model code with the LLM, without a strong feedback system and without a robust loop to push corrections back to the LLM after bad outcomes.

## Submission 2 (`sub_2`)

In this phase, we started improving training setup details:

- increased epochs
- integrated more tunable parameters

However, the experimentation and evaluation system was still not strong enough to reliably test and compare alternatives.

## Submission 3 (`sub_3`)

This phase became much more systematic.  
We explored a broader parameter space using randomized experiments and then focused on promising winners.

Main parameter dimensions explored included:

- depth
- base filters / filter scaling pattern
- learning rate
- optimizer
- dropout 
- batch normalization
- residual connections
- weight decay
- classifier hidden units
- pooling type
- batch size
- spectrogram settings (`n_mels`, `n_frames`)

After the systematic exploration, we let the AI run more freely to search for stronger alternatives.

## Submission 4 (`sub_4`)

This was a scaled-up continuation of the same idea as Submission 3.

- longer runtime (around 14 hours for the agent)
- around 160 total experiments
- final model training on full data after the search and selection steps

## Submission 5 — v1.4 (`sub_4_29.04_v1.3` + online augmentation B)

Starting from the same winning architecture and hyperparameters as Submission 4 (`submission_archive/sub_4_29.04_v1.3`), we re-ran **final full-data training** with **online augmentation** (re-sampled every batch, every epoch — not one-shot):

- mild Gaussian noise on the mel spectrogram
- small time-axis shift
- SpecAugment-style **time masking** and **frequency masking**

**Mixup was disabled** for this run.  
Artifacts were produced under `submission/submission_01/` (e.g. `model.keras`, `kaggle_inference.ipynb`) via `scripts/run_sub4_augmented_submissions.py`.
Observed score: **0.721**.

## Submission 6 — v1.5 (`sub_4_29.04_v1.3` + online augmentation C)

Same base as v1.4 (Submission 4 / `sub_4_29.04_v1.3`), with the **same online B augmentations**, plus **Mixup** during training batches (**α = 0.25**).  
Artifacts under `submission/submission_02/`, same runner script as v1.4.
Observed score: **0.720**.

## Versions 1.6 and 1.7 (secondary labels)

For **v1.6** and **v1.7**, training incorporated **secondary labels**. Despite that extra supervision signal, **neither run improved the final leaderboard AUC** compared to earlier best scores—the public score did not increase on these attempts.

## Overall Summary Across Submissions 1–4

All four submissions relied on building a convolutional neural network from scratch.

Even after running a wide range of experiments, we were not able to obtain an AUC score higher than about 0.65.

Our conclusion is that this is likely a limitation of the current concept (training from scratch with the present setup).  
To move beyond this ceiling, the agent will likely need stronger guidance and a revised strategy, because more than the current approach appears necessary to achieve a meaningfully higher AUC.

**Submissions 5–6 (v1.4 / v1.5)** keep the same CNN-from-scratch winner as Submission 4 but add **online augmentation** (and optional Mixup) in a dedicated final-training rerun, to reduce overfitting and test whether leaderboard performance improves without changing the core search pipeline.

Results for the two augmentation runs were **0.721 (v1.4)** and **0.720 (v1.5)**.  
This is a clear improvement versus the earlier non-augmentation ceiling (~0.65), and we take this as strong evidence that augmentation is useful in our setup.
