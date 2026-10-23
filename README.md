# TinyRF-KD

Research code and experimental records for **Beyond Accuracy: Multidimensional Fidelity of Tiny Neural Intrusion Detectors for WSNs**.

Nishant Harkut and Mahendra K. Shukla

Atal Bihari Vajpayee Indian Institute of Information Technology and Management, Gwalior, India

TinyRF-KD and CuKD-XAI are names for the same project. The repository contains random-forest-to-neural knowledge distillation experiments, probability and attribution analyses, fixed-point conversion tools, and microcontroller replay results.

[Reproducibility](REPRODUCIBILITY.md) · [Artifact guide](ARTIFACT.md) · [Evidence registry](results/evidence_registry/fgds_20260814_current/EVIDENCE_REGISTRY.md) · [Repository map](docs/repository/REPOSITORY_MAP.md)

## Overview

The project examines what compact intrusion detectors retain after compression: classification performance, probability estimates, agreement with a teacher, feature attributions, and numerical behavior after integer conversion. These are evaluated separately rather than treating accuracy as sufficient evidence for all of them.

A three-fold isotonic-calibrated random forest supervises two small multilayer perceptrons. Each student is compared with the same architecture trained from class labels alone. The primary experiment uses WSN-DS, a simulated wireless-sensor-network intrusion-detection dataset. Exact groups of identical input vectors remain within one partition, and preprocessing is fitted on training data only.

The study also includes ONNX/OpenVINO conversion, fixed-point C implementations, USB and Wi-Fi replay, and secondary Edge-IIoTset experiments. These results use explicitly identified checkpoints and evaluation settings; they are not pooled into one result.

## Method

1. Partition the source records by exact input-feature groups and fit the scaler on training data.
2. Train the forest teacher and paired scratch/RF-KD students over ten seeds.
3. Evaluate classification, proper probability scores, and teacher-response agreement on the held-out records.
4. Convert the student checkpoints to fixed-point representations, retaining both eligible and failed instances.
5. Compare eligible C implementations with a frozen Python integer reference, then replay selected artifacts on physical boards.

The primary RF-KD setting is temperature `T=4` and soft-loss weight `alpha=0.7`. It is a fixed experimental configuration, not a demonstrated optimum for the RF teacher.

| Model | Architecture | Trainable parameters | FP32 parameter payload |
|---|---|---:|---:|
| Student A | 17–32–16–5 | 1,189 | 4,756 B |
| Student B | 17–64–32–5 | 3,397 | 13,588 B |
| Full neural reference | 17–128–256–128–5, with BatchNorm and dropout | 69,893 | 279,572 B |

Student A and B have 58.8× and 20.6× fewer trainable parameters than the full neural reference. These ratios do not compare their storage with the forest, and parameter payload is not firmware size or measured RAM.

## Main results

### Classification

The primary split contains 262,197 training, 56,163 validation, and 56,301 test records, with 17 predictors and five classes. The table reports mean macro-F1 and sample standard deviation over ten paired training seeds on this one fixed split.

| Model | Training | Macro-F1, mean ± SD |
|---|---|---:|
| Random forest | Uncalibrated | 0.97948 ± 0.00028 |
| Random forest | Isotonic-calibrated | 0.97950 ± 0.00049 |
| Full neural reference | Class labels | 0.92195 ± 0.00170 |
| Student A | Scratch | 0.91479 ± 0.00566 |
| Student A | RF-KD | 0.91378 ± 0.00455 |
| Student B | Scratch | 0.93287 ± 0.00573 |
| Student B | RF-KD | 0.93214 ± 0.01093 |

At the primary setting, paired tests do not detect a macro-F1 improvement from RF-KD over scratch training. This is not evidence of equivalence, nor a conclusion about all temperatures, loss weights, or data splits. The students also remain below the forest's macro-F1.

The [full-route aggregate](results/wsnds/evidence_completion_20260811/fgds_controlled_full_routes_10seed_v2/aggregate_results.json) includes neural-teacher, curriculum, random-pacing, SMOTE-teacher, and co-distillation comparisons. These routes are not all compute-matched.

### Probability estimates and attributions

RF-KD has lower Brier score and negative log-likelihood than scratch for every paired seed of both students. The [probability analysis](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/aggregate_results.json) reports the per-model summaries; [paired tests](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/paired_tests.json) retain the seed-level comparisons.

This observation does not isolate RF calibration as its cause. Scratch uses class-weighted cross-entropy, whereas RF-KD combines that loss with an unweighted KL term. Both the targets and the loss balance change.

RF-KD also reduces held-out divergence from the teacher at the distillation temperature. This evaluates the objective being optimized, not an independent measure of detection utility. The [response analysis](results/wsnds/evidence_completion_20260812/fgds_behavioral_transfer_logits_10seed_v5/behavioral_transfer_summary.json) records the comparison.

A separate [SHAP analysis](results/wsnds/evidence_completion_20260811/fgds_seed42_reconstructed_teacher_shap_v3/shap_report.json) uses 500 test records and seed-42 deployment checkpoints. It reports partial feature-rank agreement, not causal explanation transfer or a ten-seed scratch-controlled attribution result.

### Integer conversion and hardware

Of the 40 primary model instances—two architectures, two training routes, and ten seeds—26 pass the numerical gates. All 26 reproduce the frozen Python integer reference exactly in native C over all 56,301 test records. The [conversion report](results/wsnds/evidence_completion_20260813/fgds_all_seed_fixed_point_audit_v1/all_seed_fixed_point_report.json) retains all 14 gate failures.

The final USB campaign replays three eligible seed-42 artifacts on an ESP32-C3 and an Arduino UNO R4 WiFi: Student A scratch, Student A RF-KD, and Student B RF-KD. Across six sessions, all 337,806 predictions and their fixed-point logits match the integer reference. Fixed macro-F1 ranges from 0.90367 to 0.91702.

This demonstrates execution agreement with the integer implementation, not equality with FP32 probabilities. The conversion policy and eligibility thresholds were developed post hoc. Board timing includes integer preprocessing and inference; serial round-trip time is reported separately.

See the [USB campaign report](results/hardware_hil/final_fgds_seed42_v1/final_campaign_usb_v1/final_hil_summary.json) for model-specific results and timing. MSP430F1611 results are [static cross-compilation measurements](deployment/msp430/current_fgds_static/artifacts/msp430_static_summary.json), not physical TelosB execution.

## Which results belong together?

| Experiment group | Use |
|---|---|
| Primary ten-seed WSN-DS | Scratch/RF-KD classification and probability comparisons, all-40 conversion screening, and selected seed-42 USB replay |
| Separate five-seed WSN-DS deployment | Seed-42 SHAP, ONNX/OpenVINO, software-only fixed-point refinement, and Wi-Fi replay |
| Secondary Edge-IIoTset protocols | Dataset-specific comparisons under separately recorded features, partitions, and training settings |

A shared label such as “Student A, seed 42” does not establish checkpoint identity across these groups. Software-refined models were not substituted for the replayed firmware. The ten training seeds are stochastic repetitions on one split, not ten independent datasets.

## Datasets

WSN-DS provides simulated normal traffic and four attack classes: Blackhole, Grayhole, Flooding, and TDMA scheduling attacks. The primary experiment removes the node identifier and retains 17 predictors. Group-disjoint partitioning prevents identical input vectors from crossing partitions; it does not establish independence between simulation scenarios or generalization to a physical network.

Edge-IIoTset is a secondary IoT/IIoT intrusion-detection dataset. Its experiment-specific preprocessing and partitions are documented separately from WSN-DS.

- WSN-DS: Almomani et al., [A Dataset for Intrusion Detection Systems in Wireless Sensor Networks](https://doi.org/10.1155/2016/4731953).
- Edge-IIoTset: Ferrag et al., [A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning](https://doi.org/10.1109/ACCESS.2022.3165809).

Dataset availability and use remain subject to the original providers' terms. Dataset files are not covered by the repository's code license.

## Environment

The reference environment uses Python 3.11. For an existing authorized local workspace, create and activate a virtual environment before installing dependencies.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Bash:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements specify version ranges, not an exact environment lock. Use the recorded environment information in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and the relevant execution manifest when reproducing an experiment. Training, runtime conversion, and board builds have different dependencies.

Some datasets and model artifacts are managed with Git LFS. The corresponding full files must be present for analyses that use them; missing files and LFS pointers are not usable dataset or checkpoint copies.

## Reviewing and reproducing results

Start with the saved reports; no training or board connection is needed to read them.

| Task | Entry point |
|---|---|
| Primary split, models, and seed-level classification results | [Ten-seed analysis](results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811/feature_group_10seed_analysis/feature_group_10seed_analysis.json) |
| Experiment inventory and artifact hashes | [Evidence registry](results/evidence_registry/fgds_20260814_current/evidence_registry.json) |
| Probability scores and reliability bins | [Probability-analysis directory](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/) |
| Hyperparameter sensitivity | [Temperature/alpha surface](results/wsnds/evidence_completion_20260811/fgds_rfkd_hyperparameter_sensitivity_10seed_v1/response_surface_aggregate.json) |
| Split sensitivity | [Ten-split summary](results/wsnds/evidence_completion_20260812/fgds_multisplit_core_10x2_v2/multisplit_core_summary.json) |
| Secondary group-aware evaluation | [Edge-IIoTset summary](results/leftover_e2e_closure/04_edge_group_aware/edge_group_aware_summary.json) |
| Training and regeneration commands | [Reproducibility guide](REPRODUCIBILITY.md) |
| Firmware preparation and board replay | [USB guide](deployment/hardware_hil/README.md), [Wi-Fi guide](deployment/wireless_hil/README.md) |

The August evidence registry and the later probability analysis are separate records. A registry status is not a claim that every later analysis has been included or that planned experiments have been completed.

After installing the dependencies, run the repository and CLI smoke checks:

```text
python -m pytest tests/repository/test_active_cli_smoke.py tests/repository/test_repository_structure_smoke.py -q
```

These checks do not train models or replay records on boards. Some checks require the referenced local evidence files. They do not replace numerical verification of the results.

Before a training rerun, inspect the driver arguments:

```text
python experiments/wsnds/leakage_free_rerun/run_feature_group_10seed_confirmation.py --help
```

The [primary experiment guide](experiments/wsnds/leakage_free_rerun/README.md) separates preflight, training, and analysis. Write reruns to a new output directory rather than replacing the retained results. Full training requires the dataset and appropriate compute; physical replay additionally requires the boards, firmware toolchains, and serial access.

## Scope and limitations

The experiments start from extracted tabular features. They do not evaluate live packet capture, on-device feature extraction, energy consumption, battery life, or variation across multiple specimens of each board.

The current evidence also does not establish a universal predictive advantage for RF-KD, causal attribution transfer, or preservation of FP32 probability quality after fixed-point conversion. Ten-seed scratch-controlled SHAP and primary-checkpoint Wi-Fi replay remain unexecuted. Existing Wi-Fi results use the separate deployment checkpoints.

Edge-IIoTset contains multiple protocols with different representations and overlap properties. Its results should not be pooled with WSN-DS or interpreted as a controlled test of grouping alone. See the [artifact guide](ARTIFACT.md) and [claim boundaries](results/evidence_registry/fgds_20260814_current/claim_boundaries.csv) for the detailed exclusions.

## Repository layout

```text
experiments/       Training and analysis programs
results/           Saved predictions, summaries, manifests, and hardware records
deployment/        Fixed-point export, firmware, host replay, and toolchain records
data/              Dataset locations and LFS-managed files
tests/             Repository, numerical, export, and replay-tool tests
manuscript/        Manuscript sources and figures
docs/              Research and repository documentation
research_history/  Earlier experiments and preserved development records
```

Historical results are retained for traceability, not as interchangeable replacements for the primary experiment.

## Citation and license

Use [CITATION.cff](CITATION.cff) for the repository's citation metadata. No journal acceptance or manuscript DOI is implied by this repository.

Original material covered by the [copyright notice](LICENSE) is **all rights reserved**. No public reuse license is granted. Use, copying, modification, or redistribution requires written permission unless applicable law, hosting-platform terms, or an existing license permits otherwise. This notice does not revoke permissions granted for earlier MIT-licensed versions.

Datasets, third-party papers, dependencies, and vendor materials retain their own terms; see [NOTICE.md](NOTICE.md). Citation does not itself grant permission to reuse restricted material. Public repository visibility is not a restriction on downloading or forking; access settings must be managed separately.

## Contact

For research questions or access enquiries, contact Nishant Harkut at [img_2023040@iiitm.ac.in](mailto:img_2023040@iiitm.ac.in). Access and reuse are governed by the repository settings and the applicable license, not by the presence or absence of setup instructions.
