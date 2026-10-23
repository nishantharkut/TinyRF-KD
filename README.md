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

## Extended results

The sections below cover the completed experiment families. Tables summarize the recorded results; linked reports retain all routes, seed-level values, uncertainty summaries, and protocol details. Primary WSN-DS, separate deployment checkpoints, and secondary datasets remain separate.

<details>
<summary>Training routes and class-level performance</summary>

| Training route | Student A macro-F1 mean (sample SD) | Student B macro-F1 mean (sample SD) |
|---|---:|---:|
| Scratch | 0.914792 (0.005658) | **0.932867 (0.005727)** |
| RF-KD | 0.913781 (0.004546) | 0.932142 (0.010930) |
| MLP-KD | 0.914321 (0.003204) | 0.922492 (0.003731) |
| Curriculum KD, legacy `fair` route ID | 0.915357 (0.003084) | 0.920344 (0.005458) |
| Curriculum KD, extended | 0.912296 (0.003330) | 0.922113 (0.005211) |
| Random-pacing KD | 0.914970 (0.003614) | 0.921007 (0.003971) |
| SMOTE-teacher KD | 0.911577 (0.006552) | 0.919619 (0.005092) |
| RF and curriculum co-distillation | **0.917463 (0.006553)** | 0.927575 (0.007727) |

Values are macro-F1 mean (sample SD) over ten seeds. The legacy `fair` identifier does not establish equal compute. Curriculum and co-distillation also change training schedules or objectives.

| Class | A scratch | A RF-KD | A delta | B scratch | B RF-KD | B delta |
|---|---:|---:|---:|---:|---:|---:|
| Blackhole | 0.862590 | 0.826224 | -0.036366 | 0.894154 | 0.872244 | -0.021910 |
| Flooding | 0.932236 | 0.941345 | +0.009109 | 0.933193 | 0.941559 | +0.008366 |
| Grayhole | 0.843567 | 0.841241 | -0.002326 | 0.896507 | 0.886806 | -0.009701 |
| Normal | 0.996226 | 0.997987 | +0.001760 | 0.997403 | 0.998128 | +0.000725 |
| TDMA | 0.939341 | 0.962106 | +0.022765 | 0.943079 | 0.961973 | +0.018893 |

Class entries are ten-seed mean F1; deltas are RF-KD minus scratch. RF-KD improves Flooding and TDMA while reducing Blackhole and Grayhole F1 in these comparisons.

Source: [full-route results](results/wsnds/evidence_completion_20260811/fgds_controlled_full_routes_10seed_v2/aggregate_results.json) and [primary class-level results](results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811/feature_group_10seed_analysis/feature_group_10seed_analysis.json).

</details>
<details>
<summary>Repeated-pattern, hyperparameter, and split sensitivity</summary>

| Evaluation view | Student A RF-KD minus scratch | Exact p | Holm p | Student B RF-KD minus scratch | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Row-level primary | -0.001012 | 0.556641 | 1.000000 | -0.000725 | 0.845703 | 1.000000 |
| Inverse exact-group-size | +0.001209 | 0.431641 | 1.000000 | -0.000204 | 1.000000 | 1.000000 |
| One representative per pure group | +0.001209 | 0.431641 | 1.000000 | -0.000204 | 1.000000 | 1.000000 |

The two group-balanced views coincide because the test groups are label-pure. They supplement, rather than replace, the row-level result.

The complete temperature/alpha grid is below. ΔF1 is RF-KD minus the persisted scratch baseline. Holm p-values use the global 18-comparison family.

| Temperature | Alpha | A: mean ΔF1 | A: global Holm p | B: mean ΔF1 | B: global Holm p |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3 | +0.009276 | 0.105469 | +0.004344 | 0.837891 |
| 1 | 0.5 | +0.005672 | 1.000000 | +0.002256 | 1.000000 |
| 1 | 0.7 | +0.010066 | 0.105469 | +0.003848 | 1.000000 |
| 2 | 0.3 | +0.004414 | 1.000000 | -0.001375 | 1.000000 |
| 2 | 0.5 | +0.006964 | 0.683594 | +0.002464 | 1.000000 |
| 2 | 0.7 | +0.006991 | 0.312500 | +0.007451 | 0.556641 |
| 4 | 0.3 | -0.002071 | 1.000000 | -0.002641 | 1.000000 |
| 4 | 0.5 | +0.000443 | 1.000000 | -0.000796 | 1.000000 |
| 4 | 0.7 | -0.001012 | 1.000000 | -0.000725 | 1.000000 |


The grid comprises 180 training jobs and reuses one test partition. It was not used to replace the fixed primary setting with a test-selected optimum.

| Student | Split-level mean RF-KD effect | Sample SD | Range | Positive / negative splits |
|---|---:|---:|---:|---:|
| Student A | +0.007126 | 0.003730 | +0.003743 to +0.016149 | 10 / 0 |
| Student B | +0.001163 | 0.004799 | -0.006576 to +0.007646 | 5 / 5 |

Each split-level value averages two paired optimizer seeds. Holdouts overlap, so these are descriptive split-sensitivity results, not independent replications. The ten-split experiment contains 80 training jobs.

Sources: [group weighting](results/wsnds/evidence_completion_20260811/fgds_group_balanced_routes_10seed_v1/aggregate_results.json), [full hyperparameter surface](results/wsnds/evidence_completion_20260811/fgds_rfkd_hyperparameter_sensitivity_10seed_v1/response_surface_aggregate.json), and [split summaries](results/wsnds/evidence_completion_20260812/fgds_multisplit_core_10x2_v2/multisplit_core_summary.json).

</details>
<details>
<summary>Probability scores and teacher-response agreement</summary>

| Student | Training | Brier score | NLL | Top-label ECE (15 bins) | Classwise ECE (15 bins) |
|---|---|---:|---:|---:|---:|
| A | Scratch | 0.033291 ± 0.002998 | 0.092596 ± 0.005760 | 0.047292 ± 0.003865 | 0.022537 ± 0.001844 |
| A | RF-KD | 0.020065 ± 0.000691 | 0.037299 ± 0.000876 | 0.003294 ± 0.000505 | 0.001744 ± 0.000260 |
| B | Scratch | 0.028261 ± 0.002015 | 0.083832 ± 0.004969 | 0.047382 ± 0.004232 | 0.021525 ± 0.001709 |
| B | RF-KD | 0.016100 ± 0.002189 | 0.031323 ± 0.002925 | 0.003185 ± 0.000667 | 0.001716 ± 0.000301 |


Values are mean ± sample SD over ten seeds. Lower is better. Brier score uses the multiclass sum over five classes. The four Brier/NLL comparisons have Holm-adjusted p = 0.0078125; ECE variants are descriptive. The change in weighted versus unweighted loss contributions prevents a causal attribution of these improvements to RF calibration alone.

| Student | Mean scratch-minus-RF-KD KL | Sample SD | Positive seeds | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Student A | 0.191178 | 0.045466 | 10 of 10 | 0.003906 |
| Student B | 0.193702 | 0.021476 | 10 of 10 | 0.003906 |

KL is exact-feature-group balanced at T=4. Lower held-out teacher divergence is consistent with the training objective; it is not an independent detection metric.

Sources: [probability summaries](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/aggregate_results.json), [paired tests](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/paired_tests.json), [per-seed metrics](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/per_seed_metrics.csv), [reliability bins](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/reliability_bins_15.csv), [classwise ECE](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/classwise_ece_15.csv), and [teacher-response report](results/wsnds/evidence_completion_20260812/fgds_behavioral_transfer_logits_10seed_v5/behavioral_transfer_summary.json).

The [analysis contract](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/analysis_contract.json), [reliability summary](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/reliability_summary_15.csv), and [artifact manifest](results/wsnds/evidence_completion_20260816/fgds_calibration_10seed_v2/artifact_manifest.json) document the supporting files.

The saved [analysis program](experiments/wsnds/evidence_completion/analyze_fgds_calibration.py) computes these metrics from preserved probabilities without training. Its [metric tests](tests/repository/test_fgds_calibration_evidence.py) cover formulas, binning, class order, and paired-seed summaries.

</details>
<details>
<summary>SHAP attribution comparison</summary>

| Output contract | Student A rank rho mean (SD) | Student B rank rho mean (SD) | Student A top-5 overlap | Student B top-5 overlap |
|---|---:|---:|---:|---:|
| FP32 probabilities, T=1 | 0.411765 (0.028897) | 0.446078 (0.016072) | 3.333 | 4.000 |
| KD-softened probabilities, T=4 | 0.403595 (0.012336) | 0.553922 (0.021226) | 3.000 | 3.667 |

This uses 500 fixed test records from the separate seed-42 deployment experiment. Standard deviations reflect three background/estimator replicates, not retrained models. The sample contains 454 Normal records and only four Flooding records. The forest was reconstructed and checked against retained train/test outputs; equivalence on SHAP-masked inputs is not established. There is no matched scratch SHAP comparator.

Source: [SHAP report](results/wsnds/evidence_completion_20260811/fgds_seed42_reconstructed_teacher_shap_v3/shap_report.json).

</details>
<details>
<summary>Software conversion and fixed-point refinement</summary>

| Student | PyTorch FP32 macro-F1 | FP32 ONNX agreement | FP32 ONNX size | Dynamic INT8 macro-F1 | INT8 agreement | INT8 ONNX size | ONNX FP32 mean, batch 1 | OpenVINO FP32 mean, batch 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Student A | 0.916076 | 1.000000 | 5,573 B | 0.909281 | 0.997069 | 5,191 B | 0.0308 ms | 0.5230 ms |
| Student B | 0.922381 | 1.000000 | 14,407 B | 0.921323 | 0.995471 | 7,547 B | 0.0419 ms | 0.1393 ms |

These are separate deployment checkpoints, not the primary USB artifacts. OpenVINO FP32 also has prediction agreement 1.0 over the 56,301 test records. Host timings use the first test record, 50 warmups, and 300 iterations; they are not MCU timings or a matched-runtime benchmark. Dynamic INT8 ONNX uses a different quantization path from the fixed-point C implementation.

| Student | Source PTQ fixed macro-F1 | Refined fixed macro-F1 | Delta | Fixed-versus-float agreement |
|---|---:|---:|---:|---:|
| Student A | 0.905694 | 0.913642 | +0.007948 | 0.997265 |
| Student B | 0.914564 | 0.921566 | +0.007002 | 0.996341 |

Refinement changes the model state. These candidates were evaluated in software only and were not exported or replayed on boards.

Sources: [runtime report](results/runtime/onnx_openvino/wsnds/fgds_seed42_exact/runtime_report.json) and [refinement report](results/wsnds/evidence_completion_20260811/fgds_fixed_point_refinement_seed42/refinement_report.json).

</details>
<details>
<summary>All-model conversion screening and USB replay</summary>

| Route | Passed | Gate-failed | Total |
|---|---:|---:|---:|
| Student A scratch | 8 | 2 | 10 |
| Student A RF-KD | 8 | 2 | 10 |
| Student B scratch | 3 | 7 | 10 |
| Student B RF-KD | 7 | 3 | 10 |
| **Total** | **26** | **14** | **40** |

The gates require at least 0.99 fixed/FP32 prediction agreement, absolute macro-F1 change at most 0.015, zero saturation, and valid arithmetic bounds. All 14 failures miss the F1-change criterion; five also miss the agreement criterion. These are numerical gate failures, not C/Python mismatches. The criteria were developed post hoc.

| Model | Fixed macro-F1 | Fixed vs FP32 prediction agreement | MCU vs fixed prediction agreement | Exact fixed logits |
|---|---:|---:|---:|---:|
| Student A scratch | 0.903667 | 0.995879 | 1.000000 | 1.000000 |
| Student A RF-KD | 0.916959 | 0.997727 | 1.000000 | 1.000000 |
| Student B RF-KD | 0.917018 | 0.996998 | 1.000000 | 1.000000 |

| Model | Board | MCU compute mean / p95 | Host USB RTT mean / p95 | Program / global bytes |
|---|---|---:|---:|---:|
| Student A scratch | ESP32-C3 | 117.855 / 124 us | 15,822.892 / 16,932 us | 281,882 / 13,592 |
| Student A RF-KD | ESP32-C3 | 117.932 / 124 us | 16,173.188 / 17,133 us | 281,874 / 13,592 |
| Student B RF-KD | ESP32-C3 | 321.116 / 328 us | 16,635.353 / 17,748 us | 284,230 / 13,592 |
| Student A scratch | Arduino UNO R4 WiFi | 302.276 / 305 us | 14,723.629 / 15,820 us | 56,512 / 7,128 |
| Student A RF-KD | Arduino UNO R4 WiFi | 302.344 / 306 us | 14,844.833 / 15,816 us | 56,528 / 7,128 |
| Student B RF-KD | Arduino UNO R4 WiFi | 793.455 / 797 us | 15,728.752 / 16,822 us | 58,864 / 7,128 |

Each board/model session replays all 56,301 test records. Timing uses a separate class-balanced 1,000-record workload repeated three times. MCU time includes preprocessing; host round-trip time includes serial and host overhead. There is one physical specimen of each board. Program bytes and compiler-reported globals do not establish measured peak RAM.

Student B scratch is excluded before replay because fixed/FP32 agreement is 0.989574 and macro-F1 change is 0.020193. It is retained in the conversion report.

Sources: [all-model conversion report](results/wsnds/evidence_completion_20260813/fgds_all_seed_fixed_point_audit_v1/all_seed_fixed_point_report.json) and [USB campaign](results/hardware_hil/final_fgds_seed42_v1/final_campaign_usb_v1/final_hil_summary.json).

</details>
<details>
<summary>Wi-Fi replay and MSP430 static compilation</summary>

| Model | Fixed macro-F1 | Fixed vs FP32 prediction agreement | MCU vs fixed prediction agreement | Exact fixed logits |
|---|---:|---:|---:|---:|
| Student A RF-KD | 0.905694 | 0.995009 | 1.000000 | 1.000000 |
| Student B RF-KD | 0.914564 | 0.994743 | 1.000000 | 1.000000 |

| Model | Board | MCU compute mean / p95 | Host UDP RTT mean / p95 | App retries | Connection RSSI |
|---|---|---:|---:|---:|---:|
| Student A RF-KD | ESP32-C3 | 162.449 / 168 us | 48,830.028 / 243,545 us | 101 | -15 dBm |
| Student B RF-KD | ESP32-C3 | 401.424 / 405 us | 9,570.774 / 36,144 us | 0 | -19 dBm |
| Student A RF-KD | Arduino UNO R4 WiFi | 300.199 / 303 us | 98,486.718 / 143,078 us | 1 | -32 dBm |
| Student B RF-KD | Arduino UNO R4 WiFi | 789.977 / 793 us | 148,823.039 / 322,525 us | 3 | -41 dBm |

Wi-Fi replay covers 225,204 full-test predictions using the separate deployment checkpoints. Retries follow application-level timeouts; they are not raw packet-loss measurements. RSSI is a connection-time observation. Transport, timing workloads, and checkpoints differ from the primary USB campaign, so the two tables are not a controlled USB-versus-Wi-Fi benchmark.

| Model | Architecture | MACs | Fixed parameter bytes | Activation estimate | Static flash load | Static RAM lower bound | Maximum single-function stack |
|---|---|---:|---:|---:|---:|---:|---:|
| Student A | 17-32-16-5 | 1,136 | 1,348 B | 140 B | 2,846 B | 8 B | 106 B |
| Student B | 17-64-32-5 | 3,296 | 3,700 B | 236 B | 5,196 B | 8 B | 202 B |

The MSP430 results are static cross-compilation at `-Os`, including integer preprocessing. The 8 B static-RAM figure is a lower bound, and maximum single-function stack is not whole-program peak stack. No physical TelosB execution, energy, or radio integration was measured.

Sources: [Wi-Fi report](results/wireless_hil/fgds_seed42/final_report_lineage_sealed_v2/wireless_hil_final_report.json) and [MSP430 summary](deployment/msp430/current_fgds_static/artifacts/msp430_static_summary.json).

</details>
<details>
<summary>Secondary Edge-IIoTset experiments</summary>

Three protocols are retained separately. All entries below are five-seed macro-F1 mean ± sample SD; the input representations, partitions, and teacher/training settings differ.

| Protocol | Student | Scratch | RF-KD |
|---|---|---:|---:|
| Strict, 43 inputs | A | 0.66459 ± 0.02001 | 0.68626 ± 0.01082 |
| Strict, 43 inputs | B | 0.68779 ± 0.00175 | 0.71111 ± 0.00252 |
| Earlier selected-capacity, 49 inputs | A | 0.80911 ± 0.00389 | 0.81184 ± 0.00317 |
| Earlier selected-capacity, 49 inputs | B | 0.81895 ± 0.00489 | 0.81966 ± 0.00263 |
| Earlier selected-capacity, 49 inputs | C | 0.82392 ± 0.00033 | 0.82438 ± 0.00091 |
| Group-aware, 40 inputs | A | 0.73353 ± 0.02100 | 0.76756 ± 0.00791 |
| Group-aware, 40 inputs | B | 0.80560 ± 0.00380 | 0.81289 ± 0.00680 |

The full matrices also retain the neural/curriculum, LightGBM, and co-distillation routes where executed. Their scores are not pooled across protocols or with WSN-DS.

The group-aware study has 1,556,588/332,240/330,373 train/validation/test rows, 15 classes, and zero pre-encoding group overlap. Encoded exact-row overlaps remain 163 train/test, 157 train/validation, and 26 validation/test. Its A/B paired RF-KD comparisons give exact p = 0.0625/0.3125. The separate 200-tree reporting forest is not the calibrated 100-tree KD teacher.

Sources: [strict full matrix](results/edge_iiot/strict_generalization/edgeiiot_v23_results.json), [49-input full matrix](results/edge_iiot/literature_comparable/edgeiiot_v23_results.json), and [group-aware summary](results/leftover_e2e_closure/04_edge_group_aware/edge_group_aware_summary.json).

</details>

Completed summaries are distinct from raw execution archives. Some large logs and bundles have separate archive locators in the saved reports. A result table does not imply that every raw build file is distributed here. Interrupted attempts, superseded experiments, compiler caches, and session transcripts are not treated as completed result evidence.

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
