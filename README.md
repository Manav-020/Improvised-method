# Subject-Specific Calibration for Explainable EEG Emotion Recognition

## Research question

**Does subject-specific calibration make EEG emotion explanations more faithful and stable to the model’s actual decision process while maintaining predictive performance?**

This project compares a general model trained on other people with a model adapted using labeled trials from one target person. Both models are evaluated on exactly the same later, untouched trials of that person. This process is repeated for all 32 people.

The implementation is in `DEAP_384_Correlation_LogReg.ipynb`. The filename is retained from an earlier version: **the current notebook extracts 128 features, not 384**.

> **Results status:** The revised notebook has passed an end-to-end synthetic-data check, but has not been run here on the user’s Windows DEAP dataset. Actual revised DEAP results are therefore pending. Historical results below belong to the previous implementation and must not be presented as results of this revision.

## 1. What the current experiment can establish

The experiment evaluates three related questions:

1. **Compact faithfulness:** Can an explanation showing five EEG channels reconstruct more of the personalized model’s response to channel masking?
2. **Refit stability:** When training and calibration trials are resampled, does the personalized model produce more consistent channel contributions on the same test examples?
3. **Predictive preservation:** Does personalization avoid a practically unacceptable decrease in balanced accuracy?

These are operational definitions. They do not establish that an explanation is physiologically correct, causal, understandable to a user, or useful in an HCI task.

For Logistic Regression, the complete interventional linear SHAP explanation is already exact in log-odds space. Personalization cannot make that calculation more exact. This notebook therefore distinguishes a full-SHAP correctness audit from the faithfulness of a short, five-channel explanation. Any resulting claim should retain that distinction.

## 2. Data and label integrity

### Dataset expected by the notebook

| Property | Expected value |
| --- | --- |
| People | 32: `s01.dat` through `s32.dat` |
| Trials per person | 40 |
| Stored channels | 40 |
| EEG channels used | First 32; peripheral channels excluded |
| Samples per trial | 8,064 at 128 Hz |
| Trial duration | 3 seconds of baseline followed by 60 seconds of stimulus |
| Target | Binary Valence |
| Low / High rule | Low: Valence ≤ 5; High: Valence > 5 |

The supplied dataset is an unofficial DEAP copy. Structural checks and matching metadata do **not** certify that its EEG signals are identical to official DEAP or independently establish EEG-to-video alignment.

### Why the labels needed attention

Earlier auditing found 163 stored ratings outside the expected range across dimensions, including 110 Valence ratings. Repairing only those out-of-range entries would be insufficient: previous comparisons also found transformed values that remained inside the valid range.

The revised notebook uses the supplied `participant_ratings.xls` reference for **every Valence label**. It does not clip values, infer a replacement scale, or overwrite the `.dat` files.

The embedded reference contains all 1,280 subject/video rows and was checked against the supplied spreadsheet. Its SHA256 is:

```text
512191a9db0814ac3042e601e5ed4a2d8a578223678d19b74b29825df9a7ea9a
```

With this reference, the complete dataset contains 572 Low-Valence and 708 High-Valence trials. The earlier approximately 79%/21% distribution should not be reused as the corrected dataset distribution. Individual calibration and test subsets can still be imbalanced.

### Video order versus presentation order

- `Participant_id` identifies the person.
- `Experiment_id` identifies the video and maps to the preprocessed trial/video position assumed by this notebook.
- `Trial` identifies when that person saw the video: its **presentation order**.

These last two identifiers must not be interchanged. Labels are matched by person and video. Chronological calibration/testing is determined by presentation order.

Stored Dominance and Liking are compared with reference values as alignment anchors. A mismatch stops the run. Passing anchors checks label-row consistency, but does not prove that the corresponding EEG signal is authentic or correctly aligned.

## 3. Professor suggestions: what was retained

| Suggestion | Decision in this revision | Reason |
| --- | --- | --- |
| Check Logistic Regression multicollinearity | Detailed diagnosis deferred, as requested | Correlation filtering and L2 regularization reduce some redundancy, but do not prove multicollinearity is resolved |
| Use 3–4 seconds of resting data as personal context | Excluded from the main analysis | The earlier baseline-relative implementation performed worse on unseen trials; a short reference can also produce noisy normalization |
| Use half of each signal for calibration and the other half for testing | Excluded from the main analysis | Both halves share a trial, label, recording conditions and possibly baseline; this measures within-trial adaptation rather than transfer to separate trials |
| Remove highly correlated inputs using a high threshold | Retained at absolute correlation > 0.90 | Removes near-duplicate features using source training data only |
| Start from the sixth second | Retained | Skip the first five seconds of stimulus after the baseline |

The earlier “384 × 348 channels” wording should be treated as a feature-matrix question, not a channel count. The current starting feature correlation matrix is **128 × 128**, because there are 128 engineered features and only 32 EEG channels.

Excluding resting normalization does not prove that all resting-state personalization methods are ineffective. It avoids making an unsuccessful earlier variant part of the primary experiment.

## 4. Step-by-step implementation

### Step 1 — Configure paths and settings

The dataset path is already configured:

```python
DATA_DIR = Path(r'D:\coding\python\HCI RESEARCH PROJECT\imlementation\data_preprocessed_python')
```

`imlementation` is the folder spelling supplied by the user. Change the path if the local directory differs.

Each run writes to a fresh timestamped directory under `deap_final_research_results/`, relative to the notebook’s working directory. This prevents results from separate runs being mixed.

### Step 2 — Audit files and assign reference labels

The loader checks that all 32 expected files exist and verifies the shapes of signals and labels. It checks EEG finiteness, rejects flat retained EEG channels, checks label anchors, and validates reference Valence. Dataset file hashes, signal audit information, and stored/reference label comparisons are saved.

Only load trusted `.dat` files: this format is read with Python pickle.

### Step 3 — Select the EEG segment

For each trial:

1. Exclude the 3-second baseline from the model inputs.
2. Skip the first 5 seconds of stimulus.
3. Retain stimulus time 5–60 seconds: approximately 55 seconds.

The sample offset is `(3 + 5) × 128 = 1024`. “Sixth second” here means the sixth **stimulus** second, not the sixth second of the complete stored recording.

### Step 4 — Extract 128 log-band-power features

Welch’s method estimates power spectral density using 256-sample windows and 128-sample overlap. Power is summed over frequency bins within each band, multiplied by frequency resolution, and log-transformed with a small numerical floor.

| Band | Frequency interval |
| --- | --- |
| Theta | 4 ≤ f < 8 Hz |
| Alpha | 8 ≤ f < 13 Hz |
| Beta | 13 ≤ f < 30 Hz |
| Gamma | 30 ≤ f < 45 Hz |

There are `32 channels × 4 bands = 128 features`. Each complete retained trial contributes **one row**. No temporal halves are treated as independent trials.

Log power reduces the influence of large raw-power values. Relative power and differential entropy are omitted to reduce overlapping feature definitions. Log transformation does not remove all subject differences, artifacts or correlations.

### Step 5 — Hold out one target person

For each target person, including person 32:

| Partition | Data | Number of trials |
| --- | --- | --- |
| Source training | All trials from the other 31 people | 1,240 |
| Target calibration | Target person’s first 20 trials in presentation order | 20 |
| Target test | Target person’s remaining 20 trials in presentation order | 20 |

**Yes, calibration is performed on the held-out person.** That person changes across the 32 outer evaluations. “Unseen trials” means separate trials from the same target person that were never used for fitting.

Partition assertions check that source, calibration and test rows are disjoint and that calibration precedes testing. A whole trial belongs to one partition only.

### Step 6 — Select settings using source people only

A three-fold `GroupKFold`, grouped by source subject, selects Logistic Regression `C` from `[0.01, 0.1, 1.0]` using balanced accuracy.

Within each inner fold, the complete pipeline is refitted:

1. Remove constant and highly correlated features.
2. Standardize retained features.
3. Fit class-balanced Logistic Regression.

This avoids fitting the filter or scaler on inner-validation data. The selected preprocessing is then fitted on all 31 source people and frozen. Target calibration and test data do not select `C`, feature retention, scaling or the SHAP background.

The correlation filter is deterministic and greedy: candidates are compared against previously retained features, and a candidate is removed if any absolute Pearson correlation exceeds 0.90. Feature order can affect which representative survives.

### Step 7 — Fit the models and controls

| Model / control | What it receives | Purpose |
| --- | --- | --- |
| General | Source trials only | Reference before target-specific calibration |
| Personalized | Source trials plus 20 labeled target calibration trials | Main adaptation method |
| Subject-only | Target calibration trials only, with the same source-derived feature space and `C` | Diagnostic of learning from target data alone |
| Majority | Source class prior | Simple prediction reference |
| Shuffled-label personalization | Source data plus randomly permuted calibration labels | Tests whether correctly matched calibration labels matter |

The personalized model is **refitted on the combined data**, rather than incrementally updated. Calibration trials receive 10 times the relative weight of a source trial. Source-derived class factors stay fixed across general and personalized fits. Total fitting weight is normalized to the source sample count so that adding calibration rows does not automatically change regularization scale through a larger total weight.

The weight of 10 is a fixed experimental choice, not a demonstrated optimum. The subject-only model is not separately tuned. If calibration contains only one class, subject-only Logistic Regression is marked unavailable; test trials are never moved into calibration to solve that problem.

Shuffled-label personalization is repeated 10 times, preserving calibration class counts. It can retain useful class-prior information, so its outcome should be measured rather than assumed to be chance.

### Step 8 — Measure prediction on untouched target trials

The decision threshold is fixed at probability 0.5. Saved outcomes include accuracy, balanced accuracy, Macro-F1, ROC-AUC and confusion-matrix counts.

Balanced accuracy is the main preservation endpoint because class counts can differ across people. If a test set contains only one class, balanced accuracy and ROC-AUC are recorded as undefined. Such cases are not repaired by rearranging the test set; valid paired counts are reported.

### Step 9 — Compute and verify channel explanations

All explanations target the same quantity: **High-Valence log odds**, even when a trial is predicted Low-Valence.

For a standardized retained feature, the interventional linear SHAP contribution is:

```text
contribution = model coefficient × (feature value − source-background mean)
```

The notebook computes this analytic formula directly; the `shap` package is not required. Feature contributions are summed, with their signs, within each EEG channel. All 32 channel positions remain present, with zero contributions where no corresponding features were retained.

Two correctness checks must pass:

- Expected log odds plus all contributions reconstruct the model output.
- Full channel contributions reconstruct actual log-odds changes under the specified channel masking.

These checks verify implementation correctness, not a personalization benefit.

### Step 10 — Evaluate compact explanation faithfulness

For each test example, select the five channels with the largest absolute signed channel contributions. These form the compact explanation.

Across 100 perturbation repetitions, each channel is independently masked with probability 0.25. Masking replaces that channel’s retained feature values with source-background means in standardized space.

Compare the model’s actual log-odds change with the change predicted using only the selected five channel contributions:

```text
compact_faithfulness = 1 − mean((actual change − predicted change)²)
                          / mean(actual change²)
```

| Score | Interpretation under this masking protocol |
| --- | --- |
| 1 | Exact reconstruction |
| 0 | Same squared error as predicting no change |
| Below 0 | Worse than predicting no change |
| Undefined | Actual response energy is too close to zero for a reliable ratio |

The score is normalized reconstruction performance, not classification accuracy and not a percentage of causal brain activity. A larger score can reflect a more concentrated decision rule or contribution cancellation.

A shuffled-channel-attribution control checks whether assigning contributions to the correct channels matters. Top, least-important and random channel deletion curves are also exported for masking 0, 1, 3, 5, 10, 20 and 32 channels. Signed probability changes need not be monotonic because important channels can support opposite classes.

### Step 11 — Evaluate stability across paired refits

The notebook performs 20 bootstrap refits per person:

1. Resample source trials within each source person and class.
2. Use the same source draw for the general and personalized refits.
3. Resample target calibration trials within class for personalization.
4. Keep preprocessing, `C`, explanation background and test examples fixed.
5. Compare each refit’s explanation with its corresponding original model’s explanation.

Stability is measured using signed attribution cosine similarity and top-five channel overlap. Zero-norm explanations are treated as undefined rather than assigned perfect stability.

This measures sensitivity to fitting-data resampling conditional on class counts and fixed preprocessing. It does not measure feature-selection instability or whether similar emotions should always use identical EEG channels. The general model is refitted too, so it does not receive an artificial stability of 1 from being frozen.

### Step 12 — Compare people, then generate the report

Refit scores are averaged within each person first. The 20 bootstrap fits are not treated as 20 independent people.

For each metric, compute the paired difference:

```text
delta = personalized score − general score
```

The notebook reports subject-level means, differences, bootstrap confidence intervals, sign-flip p-values, and counts of improved/tied/degraded people. It uses 20,000 resamples for its statistical calculations.

The two primary explanation endpoints are compact faithfulness and refit cosine stability. Their two-sided sign-flip p-values receive Holm correction. Other outcomes are secondary descriptive analyses. Sign-flip inference relies on exchangeability/symmetry under the null. Shared source training sets across LOSO folds also limit independent-subject interpretations.

Predictive preservation uses an illustrative balanced-accuracy noninferiority margin of **0.03**: the one-sided 95% lower bound for the paired difference must exceed −0.03. This margin needs substantive justification before viewing results; it must not be adjusted to produce a pass. A nonsignificant prediction difference alone does not establish equivalence or preservation.

## 5. Limitations of the previous method

| Previous choice / interpretation | Why it was insufficient for this question | Change in the revision |
| --- | --- | --- |
| Calibrate on one temporal half and test on the other half of the same trial | Shared trial identity and label can make adaptation appear much stronger than transfer to new trials; a `(trial, half)` disjointness check misses trial overlap | Use separate chronological calibration and test trials |
| Both halves share baseline context | Can introduce an additional shared trial fingerprint | Exclude baseline-relative normalization from the main analysis |
| Very large improvements under temporal-half evaluation | Do not establish generalization to separate target trials | Treat these historical numbers as within-trial diagnostic outcomes |
| Correct only obviously invalid labels | In-range transformed labels can remain wrong | Use matched reference Valence for all trials |
| Raw power plus relative power and DE | Skewness and overlapping definitions complicate coefficients and attributions | Use log-band power, source-only filtering and regularized fitting |
| Small Gaussian-noise stability near 1 | A linear model under tiny perturbations can show a ceiling effect with little practical discrimination | Compare explanations across paired model refits |
| Deletion score interpreted as proof of globally improved SHAP | Linear SHAP is already exact under the chosen assumptions; deletion also depends on model scale, feature concentration and masking | Audit full exactness separately and explicitly evaluate a compact explanation |
| General versus personalized only | Does not show whether gains depend on correctly paired target labels or whether target-only fitting suffices | Add subject-only, majority and shuffled-label controls |
| Nonsignificant performance change interpreted as equivalence | Absence of evidence for a difference is not evidence of acceptable preservation | Report an explicitly bounded exploratory noninferiority criterion |
| Positive historical conclusions reused after label/protocol changes | Different labels, splits and metrics define different experiments | Keep historical findings separate from revised results |

The initial reported faithfulness improvement of 0.0921 to 0.1936 belongs to an earlier stage. It cannot establish the outcome of the corrected-label revised experiment. Historical claims must be tied to their exact data, labels and protocol.

## 6. Results and outcomes

### 6.1 Verified implementation outcomes

The revised pipeline passed a reduced end-to-end test using synthetic data, including model fitting, chronological partition separation, reference metadata checks, correlation filtering, attribution reconstruction, paired statistics, plots and output exports. All 1,280 embedded reference rows were compared with the supplied spreadsheet. The notebook file also passed notebook-schema validation.

Synthetic testing checks execution and internal identities. It does not establish predictive performance, explanation gains, EEG authenticity, statistical calibration on DEAP, or completion of a full 32-person dataset run. Synthetic scores are not research findings and are not embedded as DEAP results.

### 6.2 Historical DEAP findings from the previous implementation

The following numbers are taken from the previously supplied `paired_summary.csv`. They use the earlier feature and explanation definitions. **They are included to explain the redesign, not to report the revised experiment.** Rounded values may not subtract exactly.

#### Separate unseen target trials

| Input mode | Metric | General | Personalized | Delta | Previous Holm-adjusted p |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_relative | balanced_accuracy | 0.473922 | 0.545878 | +0.071955 | 0.070534 |
| baseline_relative | macro_f1 | 0.449558 | 0.528631 | +0.079073 | 0.042548 |
| baseline_relative | roc_auc | 0.473759 | 0.561081 | +0.087322 | 0.013932 |
| baseline_relative | faithfulness | 0.119324 | 0.131453 | +0.012129 | 1.000000 |
| baseline_relative | stability | 0.997484 | 0.997546 | +0.000062 | 0.042548 |
| stimulus | balanced_accuracy | 0.498338 | 0.622130 | +0.123793 | 0.002289 |
| stimulus | macro_f1 | 0.366567 | 0.613782 | +0.247215 | 0.000041 |
| stimulus | roc_auc | 0.507079 | 0.702401 | +0.195322 | 0.002820 |
| stimulus | faithfulness | 0.190700 | 0.163257 | -0.027444 | 1.000000 |
| stimulus | stability | 0.997992 | 0.998094 | +0.000101 | 0.491940 |

#### Temporal halves of the same trial — diagnostic only

| Input mode | Metric | General | Personalized | Delta | Previous Holm-adjusted p |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_relative | balanced_accuracy | 0.488168 | 0.877299 | +0.389131 | 0.000014 |
| baseline_relative | macro_f1 | 0.466119 | 0.875707 | +0.409588 | 0.000000 |
| baseline_relative | roc_auc | 0.478984 | 0.949708 | +0.470723 | 0.000014 |
| baseline_relative | faithfulness | 0.123848 | 0.155632 | +0.031783 | 0.534556 |
| baseline_relative | stability | 0.997565 | 0.997692 | +0.000127 | 0.010594 |
| stimulus | balanced_accuracy | 0.497830 | 0.690257 | +0.192426 | 0.000000 |
| stimulus | macro_f1 | 0.369313 | 0.685737 | +0.316424 | 0.000000 |
| stimulus | roc_auc | 0.503757 | 0.739865 | +0.236108 | 0.000016 |
| stimulus | faithfulness | 0.194442 | 0.133656 | -0.060786 | 0.693689 |
| stimulus | stability | 0.997967 | 0.998201 | +0.000234 | 0.002104 |

#### Interpretation of those historical findings

1. **Prediction improved most convincingly with stimulus-only inputs on unseen trials.** Balanced accuracy rose from about 0.498 to 0.622, Macro-F1 from 0.367 to 0.614, and ROC-AUC from 0.507 to 0.702. Their previous adjusted p-values were below 0.05. This supports further investigation of target-person adaptation in that protocol; it does not establish the revised experiment’s performance.
2. **The old faithfulness result did not support a personalization benefit.** Stimulus-only unseen-trial faithfulness decreased from approximately 0.191 to 0.163, with adjusted p = 1.0. Baseline-relative faithfulness increased from approximately 0.119 to 0.131, also with adjusted p = 1.0. Its paired difference confidence interval included zero. A higher predictive score therefore did not automatically produce a more faithful explanation by the old measurement.
3. **Old stability was near a ceiling.** Values around 0.997–0.998 leave little room to distinguish methods. Some baseline-relative changes were statistically significant under the previous analysis, but very small in absolute size. Stimulus-only unseen-trial stability was not significant after correction. This motivated the refit-based assessment.
4. **Baseline-relative inputs were weaker on unseen-trial prediction.** Personalized balanced accuracy was approximately 0.546 versus 0.622 for stimulus-only inputs. This comparison is evidence against making that particular resting-normalization variant the default; it is not evidence that all baseline context is useless.
5. **Temporal-half results were much stronger, especially with baseline-relative features.** Balanced accuracy near 0.877 and AUC near 0.950 concern later segments of already-calibrated trials. They cannot be advertised as performance on separate unseen trials.
6. **General-model performance remained a concern.** Near-chance AUC means cross-subject predictive signal was weak in that implementation. More faithful explanations alone would not make an uninformative emotion classifier useful. Dataset alignment, feature quality and cross-subject variability remain possible contributors.

These findings support a careful redesign, not a confirmed answer of “yes” to the research question.

### 6.3 Revised experiment: actual results pending

| Endpoint | General | Personalized | Paired difference / uncertainty | Status |
| --- | --- | --- | --- | --- |
| Balanced accuracy | Pending local run | Pending local run | Pending | Not established |
| Macro-F1 | Pending local run | Pending local run | Pending | Not established |
| ROC-AUC | Pending local run | Pending local run | Pending | Not established |
| Compact five-channel faithfulness | Pending local run | Pending local run | Pending adjusted test and CI | Not established |
| Refit cosine stability | Pending local run | Pending local run | Pending adjusted test and CI | Not established |
| Refit top-five overlap | Pending local run | Pending local run | Pending descriptive comparison | Not established |
| Correct-label versus shuffled-label calibration | Pending local run | Pending local run | Pending control comparison | Not established |
| Prediction preservation within 0.03 BA margin | — | — | Pending one-sided lower bound | Not established |

Do not populate this table using historical scores: the features, split and explanation metrics changed. Revised compact-faithfulness scores are not directly comparable with the old deletion-based faithfulness values.

### 6.4 How to interpret the revised outcome

| Observed revised outcome | Defensible interpretation |
| --- | --- |
| Compact faithfulness and refit stability improve, with predictive preservation | Supports the operational hypothesis under this model, split, masking intervention and stability definition |
| Compact faithfulness improves but stability does not | Partial support for compact explanatory coverage; no supported stability gain |
| Stability improves but faithfulness does not | Partial support for refit consistency; no supported compact-faithfulness gain |
| Prediction improves but explanation endpoints do not | Personalization helps prediction, but the explanation-improvement hypothesis is unsupported |
| Explanation endpoints improve but prediction degrades beyond the chosen margin | The combined research objective is not met |
| Differences are small or uncertain | Inconclusive; absence of significance is not proof of equality |
| Correct-label and shuffled-label calibration behave similarly | Benefits may reflect class proportions, weighting or distribution changes; label-specific adaptation is not established |
| General and personalized models both predict poorly | Explainability differences need cautious interpretation; they do not rescue weak emotion prediction |

The notebook’s `findings.md` reports endpoint flags separately. It is not a complete scientific verdict: inspect effect sizes, valid-subject counts, uncertainty, absolute predictive quality, controls and limitations before writing the final conclusion.

A suitable reporting template after running is:

> Across [N] valid target people, personalization changed compact explanation faithfulness by [delta, 95% CI] and refit cosine stability by [delta, 95% CI]. Holm-adjusted p-values were [p] and [p]. Balanced accuracy changed by [delta], with a one-sided 95% lower bound of [bound]. The prespecified/justified preservation margin was [margin]. Correct-label calibration compared with shuffled-label calibration as follows: [control result]. These exploratory findings [support / partially support / do not support / are inconclusive for] the operational hypothesis, conditional on the stated protocol.

Do not describe the illustrative margin as prespecified unless it was actually accepted before inspecting this run’s outcomes.

## 7. How to run

1. Put `DEAP_384_Correlation_LogReg.ipynb` in the project directory.
2. Ensure `DATA_DIR` contains `s01.dat` through `s32.dat` with the expected structure.
3. Open the notebook in Jupyter or VS Code using a Python kernel.
4. If needed, install dependencies in that kernel’s environment:

   ```bash
   python -m pip install numpy pandas scipy scikit-learn matplotlib joblib threadpoolctl ipython jupyter
   ```

5. Check the path and configuration. The metadata reference is embedded, so a separate Excel file or Excel reader is not needed to execute the notebook.
6. Select **Restart Kernel → Run All**.
7. If a label-anchor, shape, finite-value, flat-channel or convergence check fails, investigate that error rather than bypassing it.
8. Wait for the per-person completion messages. The full run performs nested model selection, 10 shuffled-label fits and 20 paired bootstrap refits per person. Runtime depends on the computer; no measured full-DEAP runtime is available here.
9. Inspect the new timestamped output directory. The notebook writes a metrics checkpoint, but does not implement automatic resume.
10. Save the executed notebook and use the exported files to write the actual results section.

## 8. Output files

All paths below are relative to the timestamped run directory.

| File / location | Contents and use |
| --- | --- |
| `dataset_manifest.csv` | Original `.dat` filenames, byte sizes and hashes |
| `signal_audit.csv` | Subject-level shape, dtype, EEG finite checks and anchor status |
| `stored_reference_labels.csv` | Stored and reference ratings, matched by person/video |
| `trial_metadata.csv` | Subject, video, presentation order, corrected Valence and binary target |
| `log_power_features.csv` | 128 extracted features per trial; rows align with `trial_metadata.csv` |
| `split_manifest.csv` | Source/calibration/test membership for every target-person evaluation |
| `subject_metrics.csv` | Prediction and explanation metrics per person/model; refit stability averaged within person |
| `paired_summary.csv` | Paired deltas, confidence intervals, sign-flip tests, adjusted primary tests and valid counts |
| `trial_predictions.csv` | Test labels and High-Valence probabilities per person/model |
| `refit_stability.csv` | Individual bootstrap-refit cosine and overlap measurements |
| `shuffled_label_controls.csv` | Repeated shuffled-calibration outcomes and number of changed labels |
| `correct_vs_shuffled_calibration.csv` | Correct-label personalized outcomes alongside each person’s mean shuffled-label outcomes |
| `deletion_curves.csv` | Top/least/random channel deletion summaries in log odds and probability |
| `model_status.csv` | Unavailable model cases, such as one-class calibration for subject-only fitting |
| `metrics_checkpoint.csv` | Prediction/explanation metrics accumulated during the run; final stability merge is in `subject_metrics.csv` |
| `configuration.json` | Main experiment settings, metadata provenance and package versions |
| `findings.md` | Automatically generated exploratory endpoint statements and summary table |
| `paired_*.png`, `deletion_*.png` | Paired subject plots and deletion curves |
| `subject_XX/source_cv.csv` | Source-only hyperparameter search results |
| `subject_XX/retained_features.csv` | Source-derived retained feature list |
| `subject_XX/*_channel_attributions.csv` | Signed 32-channel contributions for test trials |
| `subject_XX/*.joblib` | Fitted model, preprocessing and explanation context for reuse |

The saved models contain a custom `CorrelationFilter`; define its notebook class before loading them in another session. Only load trusted pickle/joblib files.

For a findings review, share `paired_summary.csv`, `subject_metrics.csv`, `correct_vs_shuffled_calibration.csv`, `refit_stability.csv`, `configuration.json`, `findings.md` and `signal_audit.csv`. Plots help interpretation. Raw EEG and saved model objects are not needed for an initial review.

## 9. Limitations of the revised method

1. **Unofficial signals remain unverified.** Metadata repair cannot reconstruct unknown changes to EEG recordings or prove EEG/video alignment. Hashes identify the current files; they do not certify equivalence to official DEAP.
2. **The main faithfulness claim is deliberately narrow.** It concerns compact explanatory coverage under a defined masking intervention. It does not show improved full-SHAP exactness, physiological truth or causal importance.
3. **Mean masking can be unrealistic.** Correlated EEG features can produce implausible combinations when channels are replaced by background values. Filtering at 0.90 does not remove all dependencies.
4. **Signed grouping can hide cancellation.** Contributions from bands within one channel can oppose one another. The summed channel view is a reporting choice, not a unique physiological explanation.
5. **Stability is conditional on frozen preprocessing.** Bootstrap evaluation does not capture uncertainty from refitting feature selection, scaling, hyperparameter tuning or the explanation background. Class-stratified resampling also keeps observed class composition fixed.
6. **Twenty calibration trials is a substantial labeling budget.** It may not represent a practical quick-calibration scenario. Budget studies should use a fixed untouched test set and avoid selecting settings from its results.
7. **One chronological split does not cover every deployment setting.** It tests earlier-to-later adaptation within the recorded session. It does not establish generalization across days, devices or recording sessions.
8. **Models share training people.** LOSO outcome dependence limits simple independent-subject inference. Bootstrap intervals and sign-flip tests should be treated as exploratory evidence.
9. **Some choices are fixed rather than optimized.** Calibration weight 10, four bands, top five channels and the masking scheme are design choices. They must not be changed retrospectively to obtain favorable significance.
10. **Prediction preservation is not yet demonstrated.** The 0.03 margin is illustrative and needs justification. A positive endpoint flag is not proof of practically useful prediction.
11. **Controls have limits.** Subject-only fitting is not separately tuned. Shuffled calibration retains class counts and target features. These controls improve interpretation but do not isolate every mechanism of personalization.
12. **Current model scope is linear.** The study cannot establish that the same result holds for nonlinear classifiers. Regularization and correlation filtering do not replace the deferred multicollinearity investigation.
13. **Current sample size is small.** There are 32 people and only 20 test trials per target person. Subject metrics can be noisy or undefined for one-class test sets.
14. **The redesign is exploratory.** Earlier observed results informed these decisions. New resampling does not create an independent confirmatory dataset.
15. **No human explanation evaluation is included.** An HCI claim about clarity, trust or usefulness needs a separate user study. Computational faithfulness and stability are only part of explainability.

## 10. Next research steps after the main run

First verify audits, prediction quality, primary endpoint uncertainty and control behavior. Report a null or mixed explanation outcome honestly if that is what the experiment finds.

If extending the study, specify the extension before using test results to choose among methods. Useful directions include a lower calibration budget, stronger artifact/alignment checks, full-pipeline stability, a carefully specified nonlinear classifier with an appropriate explanation method, and an HCI study of explanation usefulness. A nonlinear model is not automatically more explainable or guaranteed to improve this research outcome.

## 11. References and methodological context

- [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) — dataset description and access information; current files remain an unofficial copy.
- [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) — fit preprocessing on training data and use pipelines to prevent leakage.
- [SHAP: LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html) — analytic interventional linear SHAP and the distinction from correlation-dependent explanations.

**Current project outcome:** a runnable, synthetic-tested exploratory evaluation with clearer trial separation, verified reference labels, controls, and explicit explanation definitions. The research question’s empirical answer remains pending the revised DEAP run.
