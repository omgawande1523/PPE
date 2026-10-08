### Promotion log (results/promotions.csv)

| run | scenario | decision | golden mAP50 cand / inc | reason |
|---|---|---|---|---|
| p5_camera_fault_20261008T004548Z | camera_fault | no_retrain | - / - | camera fault (degradation, nonfunctional, stuck_at) in windows [4, 5, 7, 8, 10, 11, 13, 14]: human alerted, retrain suppressed; temporal alarm in windows [2, 3, 15] recovered: logged and waited |
| p5_transient_20261008T005019Z | transient | no_retrain | - / - | temporal alarm in windows [2, 3, 5, 6] recovered: logged and waited |
| p5_site_shift_mild_20261008T005323Z | site_shift_mild | no_retrain | - / - | temporal alarm in windows [2, 3, 9, 13] recovered: logged and waited |
| p5_site_shift_20261008T005930Z | site_shift | rejected | 0.548983 / 0.570157 | failed: FAIL golden recall no_goggle: 0.2121 vs 0.3030 (7/33 vs 10/33) | FAIL golden recall no_gloves: 0.1552 vs 0.1897 (9/58 vs 11/58) | FAIL golden mAP50 0.5490 vs 0.5702 (drop +0.0212, tolerance 0.02) || passed: PASS golden recall no_helmet: 0.2250 vs 0.2250 (9/40 vs 9/40) | PASS golden recall no_boots: 0.0000 vs 0.0000 (0/23 vs 0/23) | PASS new-site recall 0.6382 vs 0.2287 must improve |
| p5_poisoned_20261008T013823Z | poisoned_pseudo_labels | rejected | 0.552071 / 0.570157 | failed: FAIL golden recall no_goggle: 0.2424 vs 0.3030 (8/33 vs 10/33) || passed: PASS golden recall no_helmet: 0.2250 vs 0.2250 (9/40 vs 9/40) | PASS golden recall no_gloves: 0.1897 vs 0.1897 (11/58 vs 11/58) | PASS golden recall no_boots: 0.0000 vs 0.0000 (0/23 vs 0/23) | PASS golden mAP50 0.5521 vs 0.5702 (drop +0.0181, tolerance 0.02) | PASS new-site recall 0.6288 vs 0.2287 must improve |

### p5_camera_fault_20261008T004548Z

| window | segment | est drop | true drop | true drop (no_*) | fault share | alarm |
|---|---|---|---|---|---|---|
| 00 | clean | 0.010995 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 01 | clean | -0.034113 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 02 | clean | 0.363687 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 03 | clean | 0.341810 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 04 | fault_black | 1.140045 | 0.906977 | 0.760000 | 1.000000 | 1.000000 |
| 05 | fault_black | 1.590563 | 0.877023 | 0.804348 | 1.000000 | 1.000000 |
| 06 | clean | 0.206970 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 07 | fault_covered | 1.553402 | 0.882353 | 0.730769 | 0.975000 | 1.000000 |
| 08 | fault_covered | 1.341411 | 0.861893 | 0.555556 | 1.000000 | 1.000000 |
| 09 | clean | 0.190315 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 10 | fault_frozen | 0.264716 | 0.000000 | 0.000000 | 0.975000 | 1.000000 |
| 11 | fault_frozen | 0.264716 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| 12 | clean | 0.116951 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 13 | fault_blur | 0.330975 | 0.509859 | 0.500000 | 1.000000 | 1.000000 |
| 14 | fault_blur | 0.417998 | 0.485465 | 0.459459 | 1.000000 | 1.000000 |
| 15 | clean | 0.433497 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 16 | clean | 0.041672 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 17 | clean | -0.001724 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
- val clean all monitor_alarm_threshold: 0.223440 (n=200)
- train stream all retrains: 0 (n=18)

### p5_transient_20261008T005019Z

| window | segment | est drop | true drop | true drop (no_*) | fault share | alarm |
|---|---|---|---|---|---|---|
| 00 | clean | 0.010995 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 01 | clean | -0.034113 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 02 | clean | 0.363687 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 03 | clean | 0.341810 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 04 | clean | 0.124398 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 05 | rain_s3 | 0.224036 | 0.423948 | 0.652174 | 0.000000 | 1.000000 |
| 06 | rain_s3 | 0.250769 | 0.424658 | 0.586207 | 0.000000 | 1.000000 |
| 07 | clean | -0.000600 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 08 | clean | 0.024100 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 09 | clean | 0.190315 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 10 | clean | 0.026139 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | clean | 0.014781 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 12 | clean | 0.116951 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
- val clean all monitor_alarm_threshold: 0.223440 (n=200)
- train stream all retrains: 0 (n=13)

### p5_site_shift_mild_20261008T005323Z

| window | segment | est drop | true drop | true drop (no_*) | fault share | alarm |
|---|---|---|---|---|---|---|
| 00 | clean | 0.010995 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 01 | clean | -0.034113 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 02 | clean | 0.363687 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 03 | clean | 0.341810 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 04 | clean | 0.124398 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 05 | low_light_s2 | 0.214257 | 0.171521 | 0.195652 | 0.000000 | 0.000000 |
| 06 | low_light_s2 | 0.095557 | 0.181507 | 0.310345 | 0.000000 | 0.000000 |
| 07 | low_light_s2 | 0.135117 | 0.153595 | 0.217949 | 0.000000 | 0.000000 |
| 08 | low_light_s2 | 0.122630 | 0.202046 | 0.190476 | 0.000000 | 0.000000 |
| 09 | low_light_s2 | 0.249144 | 0.188153 | 0.222222 | 0.000000 | 1.000000 |
| 10 | low_light_s2 | 0.173322 | 0.257396 | 0.267606 | 0.000000 | 0.000000 |
| 11 | low_light_s2 | 0.130571 | 0.259777 | 0.166667 | 0.000000 | 0.000000 |
| 12 | low_light_s2 | 0.070741 | 0.198083 | 0.115385 | 0.000000 | 0.000000 |
| 13 | low_light_s2 | 0.230887 | 0.205634 | 0.214286 | 0.000000 | 1.000000 |
| 14 | low_light_s2 | 0.182160 | 0.203488 | 0.324324 | 0.000000 | 0.000000 |
| 15 | low_light_s2 | 0.147635 | 0.187970 | 0.133333 | 0.000000 | 0.000000 |
| 16 | low_light_s2 | 0.182823 | 0.192568 | 0.171429 | 0.000000 | 0.000000 |
| 17 | low_light_s2 | 0.149787 | 0.122507 | 0.152542 | 0.000000 | 0.000000 |
| 18 | low_light_s2 | 0.139482 | 0.178082 | 0.136364 | 0.000000 | 0.000000 |
| 19 | low_light_s2 | 0.077833 | 0.164557 | 0.068966 | 0.000000 | 0.000000 |
| 20 | low_light_s2 | 0.170145 | 0.163978 | 0.190476 | 0.000000 | 0.000000 |
| 21 | low_light_s2 | 0.203420 | 0.144543 | 0.156250 | 0.000000 | 0.000000 |
| 22 | low_light_s2 | 0.219316 | 0.189759 | 0.279070 | 0.000000 | 0.000000 |
| 23 | low_light_s2 | 0.217677 | 0.215873 | 0.225806 | 0.000000 | 0.000000 |
| 24 | low_light_s2 | 0.148843 | 0.250000 | 0.109375 | 0.000000 | 0.000000 |
- val clean all monitor_alarm_threshold: 0.223440 (n=200)
- train stream all retrains: 0 (n=25)

### p5_site_shift_20261008T005930Z

| window | segment | est drop | true drop | true drop (no_*) | fault share | alarm |
|---|---|---|---|---|---|---|
| 00 | clean | 0.010995 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 01 | clean | -0.034113 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 02 | clean | 0.363687 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 03 | clean | 0.341810 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 04 | clean | 0.124398 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 05 | low_light_s3 | 0.409375 | 0.627832 | 0.565217 | 0.000000 | 1.000000 |
| 06 | low_light_s3 | 0.422059 | 0.660959 | 0.655172 | 0.000000 | 1.000000 |
| 07 | low_light_s3 | 0.282637 | 0.575163 | 0.641026 | 0.000000 | 1.000000 |
| 08 | low_light_s3 | 0.349251 | 0.644501 | 0.523810 | 0.000000 | 1.000000 |
| 09 | low_light_s3 | 0.394096 | 0.672474 | 0.481481 | 0.000000 | 1.000000 |
| 10 | low_light_s3 | 0.401339 | 0.627219 | 0.464789 | 0.000000 | 1.000000 |
| 11 | low_light_s3 | 0.399859 | 0.639665 | 0.483333 | 0.000000 | 1.000000 |
| 12 | low_light_s3 | 0.364175 | 0.619808 | 0.500000 | 0.000000 | 1.000000 |
| 13 | low_light_s3 | 0.405550 | 0.673239 | 0.595238 | 0.000000 | 1.000000 |
| 14 | low_light_s3 | 0.357323 | 0.694767 | 0.567568 | 0.000000 | 1.000000 |
| 15 | low_light_s3 | 0.493175 | 0.691729 | 0.633333 | 0.000000 | 1.000000 |
| 16 | low_light_s3 | 0.332905 | 0.665541 | 0.457143 | 0.000000 | 1.000000 |
| 17 | low_light_s3 | 0.355339 | 0.641026 | 0.474576 | 0.000000 | 1.000000 |
| 18 | low_light_s3 | 0.301838 | 0.571918 | 0.431818 | 0.000000 | 1.000000 |
| 19 | low_light_s3 | 0.438372 | 0.572785 | 0.344828 | 0.000000 | 1.000000 |
| 20 | low_light_s3 | 0.410814 | 0.669355 | 0.666667 | 0.000000 | 1.000000 |
| 21 | low_light_s3 | 0.436416 | 0.575221 | 0.359375 | 0.000000 | 1.000000 |
| 22 | low_light_s3 | 0.364703 | 0.650602 | 0.604651 | 0.025000 | 1.000000 |
| 23 | low_light_s3 | 0.358949 | 0.704762 | 0.677419 | 0.000000 | 1.000000 |
| 24 | low_light_s3 | 0.378454 | 0.642857 | 0.390625 | 0.000000 | 1.000000 |

| set | class | incumbent | candidate | n |
|---|---|---|---|---|
| golden_v1 clean | all | 0.715428 | 0.713030 | 1251 |
| golden_v1 clean | boots | 0.677725 | 0.682464 | 211 |
| golden_v1 clean | gloves | 0.760736 | 0.742331 | 163 |
| golden_v1 clean | goggles | 0.750000 | 0.730769 | 52 |
| golden_v1 clean | helmet | 0.921875 | 0.916667 | 192 |
| golden_v1 clean | no_boots | 0.000000 | 0.000000 | 23 |
| golden_v1 clean | no_gloves | 0.189655 | 0.155172 | 58 |
| golden_v1 clean | no_goggle | 0.303030 | 0.212121 | 33 |
| golden_v1 clean | no_helmet | 0.225000 | 0.225000 | 40 |
| golden_v1 clean | none | 0.476923 | 0.507692 | 65 |
| golden_v1 clean | person | 0.817797 | 0.830508 | 236 |
| golden_v1 clean | vest | 0.887640 | 0.893258 | 178 |
| golden_v1 clean | violation_pooled | 0.194805 | 0.162338 | 154 |
| golden_v1 low_light_s3 | all | 0.227818 | 0.611511 | 1251 |
| golden_v1 low_light_s3 | boots | 0.165877 | 0.507109 | 211 |
| golden_v1 low_light_s3 | gloves | 0.085890 | 0.539877 | 163 |
| golden_v1 low_light_s3 | goggles | 0.000000 | 0.576923 | 52 |
| golden_v1 low_light_s3 | helmet | 0.067708 | 0.796875 | 192 |
| golden_v1 low_light_s3 | no_boots | 0.000000 | 0.000000 | 23 |
| golden_v1 low_light_s3 | no_gloves | 0.103448 | 0.224138 | 58 |
| golden_v1 low_light_s3 | no_goggle | 0.090909 | 0.151515 | 33 |
| golden_v1 low_light_s3 | no_helmet | 0.250000 | 0.275000 | 40 |
| golden_v1 low_light_s3 | none | 0.276923 | 0.476923 | 65 |
| golden_v1 low_light_s3 | person | 0.563559 | 0.771186 | 236 |
| golden_v1 low_light_s3 | vest | 0.297753 | 0.814607 | 178 |
| golden_v1 low_light_s3 | violation_pooled | 0.123377 | 0.188312 | 154 |
| val low_light_s3 | all | 0.228669 | 0.638225 | 1172 |
| val low_light_s3 | boots | 0.125828 | 0.635762 | 151 |
| val low_light_s3 | gloves | 0.102941 | 0.551471 | 136 |
| val low_light_s3 | goggles | 0.021277 | 0.595745 | 47 |
| val low_light_s3 | helmet | 0.054726 | 0.736318 | 201 |
| val low_light_s3 | no_boots | 0.000000 | 0.000000 | 4 |
| val low_light_s3 | no_gloves | 0.107143 | 0.196429 | 56 |
| val low_light_s3 | no_goggle | 0.000000 | 0.073171 | 41 |
| val low_light_s3 | no_helmet | 0.133333 | 0.355556 | 45 |
| val low_light_s3 | none | 0.345679 | 0.493827 | 81 |
| val low_light_s3 | person | 0.581590 | 0.853556 | 239 |
| val low_light_s3 | vest | 0.257310 | 0.742690 | 171 |
| val low_light_s3 | violation_pooled | 0.082192 | 0.205479 | 146 |
- val clean all monitor_alarm_threshold: 0.223440 (n=200)
- train stream all retrain_trigger_window: 7 (n=1)
- train stream all frames_pooled: 400 (n=400)
- train stream all frames_mined: 200 (n=400)
- train stream all replay_frames: 200 (n=332)
- train stream all pseudo_label_error_rate_vs_source: 0.000000 (n=1547)
- train stream all pseudo_label_miss_rate_vs_source: 0.000000 (n=1547)
- train stream violation pseudo_label_error_rate_vs_source: 0.000000 (n=88)
- train stream violation pseudo_label_miss_rate_vs_source: 0.000000 (n=88)
- train stream all poisoned_boxes: 0 (n=1547)
- train stream all human_minutes: 0 (n=20)
- train stream all finetune_seconds: 1433.088892 (n=400)
- golden_v1 clean all map50_incumbent: 0.570157 (n=141)
- golden_v1 clean all map50_95_incumbent: 0.290262 (n=141)
- val low_light_s3 all map50_incumbent: 0.253518 (n=143)
- val low_light_s3 all map50_95_incumbent: 0.108700 (n=143)
- golden_v1 low_light_s3 all map50_incumbent: 0.258720 (n=141)
- golden_v1 low_light_s3 all map50_95_incumbent: 0.113625 (n=141)
- golden_v1 clean all map50_candidate: 0.548983 (n=141)
- golden_v1 clean all map50_95_candidate: 0.279202 (n=141)
- val low_light_s3 all map50_candidate: 0.493035 (n=143)
- val low_light_s3 all map50_95_candidate: 0.234477 (n=143)
- golden_v1 low_light_s3 all map50_candidate: 0.476880 (n=141)
- golden_v1 low_light_s3 all map50_95_candidate: 0.233544 (n=141)
- golden_v1 clean all promoted: 0.000000 (n=1)
- train stream all improve_seconds: 1952.872154 (n=400)
- train stream all retrains: 1 (n=25)

### p5_poisoned_20261008T013823Z

| window | segment | est drop | true drop | true drop (no_*) | fault share | alarm |
|---|---|---|---|---|---|---|
| 00 | clean | 0.010995 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 01 | clean | -0.034113 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 02 | clean | 0.363687 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 03 | clean | 0.341810 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| 04 | clean | 0.124398 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 05 | low_light_s3 | 0.409375 | 0.627832 | 0.565217 | 0.000000 | 1.000000 |
| 06 | low_light_s3 | 0.422059 | 0.660959 | 0.655172 | 0.000000 | 1.000000 |
| 07 | low_light_s3 | 0.282637 | 0.575163 | 0.641026 | 0.000000 | 1.000000 |
| 08 | low_light_s3 | 0.349251 | 0.644501 | 0.523810 | 0.000000 | 1.000000 |
| 09 | low_light_s3 | 0.394096 | 0.672474 | 0.481481 | 0.000000 | 1.000000 |
| 10 | low_light_s3 | 0.401339 | 0.627219 | 0.464789 | 0.000000 | 1.000000 |
| 11 | low_light_s3 | 0.399859 | 0.639665 | 0.483333 | 0.000000 | 1.000000 |
| 12 | low_light_s3 | 0.364175 | 0.619808 | 0.500000 | 0.000000 | 1.000000 |
| 13 | low_light_s3 | 0.405550 | 0.673239 | 0.595238 | 0.000000 | 1.000000 |
| 14 | low_light_s3 | 0.357323 | 0.694767 | 0.567568 | 0.000000 | 1.000000 |
| 15 | low_light_s3 | 0.493175 | 0.691729 | 0.633333 | 0.000000 | 1.000000 |
| 16 | low_light_s3 | 0.332905 | 0.665541 | 0.457143 | 0.000000 | 1.000000 |
| 17 | low_light_s3 | 0.355339 | 0.641026 | 0.474576 | 0.000000 | 1.000000 |
| 18 | low_light_s3 | 0.301838 | 0.571918 | 0.431818 | 0.000000 | 1.000000 |
| 19 | low_light_s3 | 0.438372 | 0.572785 | 0.344828 | 0.000000 | 1.000000 |
| 20 | low_light_s3 | 0.410814 | 0.669355 | 0.666667 | 0.000000 | 1.000000 |
| 21 | low_light_s3 | 0.436416 | 0.575221 | 0.359375 | 0.000000 | 1.000000 |
| 22 | low_light_s3 | 0.364703 | 0.650602 | 0.604651 | 0.025000 | 1.000000 |
| 23 | low_light_s3 | 0.358949 | 0.704762 | 0.677419 | 0.000000 | 1.000000 |
| 24 | low_light_s3 | 0.378454 | 0.642857 | 0.390625 | 0.000000 | 1.000000 |

| set | class | incumbent | candidate | n |
|---|---|---|---|---|
| golden_v1 clean | all | 0.715428 | 0.711431 | 1251 |
| golden_v1 clean | boots | 0.677725 | 0.672986 | 211 |
| golden_v1 clean | gloves | 0.760736 | 0.742331 | 163 |
| golden_v1 clean | goggles | 0.750000 | 0.711538 | 52 |
| golden_v1 clean | helmet | 0.921875 | 0.916667 | 192 |
| golden_v1 clean | no_boots | 0.000000 | 0.000000 | 23 |
| golden_v1 clean | no_gloves | 0.189655 | 0.189655 | 58 |
| golden_v1 clean | no_goggle | 0.303030 | 0.242424 | 33 |
| golden_v1 clean | no_helmet | 0.225000 | 0.225000 | 40 |
| golden_v1 clean | none | 0.476923 | 0.523077 | 65 |
| golden_v1 clean | person | 0.817797 | 0.826271 | 236 |
| golden_v1 clean | vest | 0.887640 | 0.882022 | 178 |
| golden_v1 clean | violation_pooled | 0.194805 | 0.181818 | 154 |
| golden_v1 low_light_s3 | all | 0.227818 | 0.601119 | 1251 |
| golden_v1 low_light_s3 | boots | 0.165877 | 0.502370 | 211 |
| golden_v1 low_light_s3 | gloves | 0.085890 | 0.533742 | 163 |
| golden_v1 low_light_s3 | goggles | 0.000000 | 0.557692 | 52 |
| golden_v1 low_light_s3 | helmet | 0.067708 | 0.807292 | 192 |
| golden_v1 low_light_s3 | no_boots | 0.000000 | 0.000000 | 23 |
| golden_v1 low_light_s3 | no_gloves | 0.103448 | 0.120690 | 58 |
| golden_v1 low_light_s3 | no_goggle | 0.090909 | 0.090909 | 33 |
| golden_v1 low_light_s3 | no_helmet | 0.250000 | 0.150000 | 40 |
| golden_v1 low_light_s3 | none | 0.276923 | 0.507692 | 65 |
| golden_v1 low_light_s3 | person | 0.563559 | 0.771186 | 236 |
| golden_v1 low_light_s3 | vest | 0.297753 | 0.808989 | 178 |
| golden_v1 low_light_s3 | violation_pooled | 0.123377 | 0.103896 | 154 |
| val low_light_s3 | all | 0.228669 | 0.628840 | 1172 |
| val low_light_s3 | boots | 0.125828 | 0.622517 | 151 |
| val low_light_s3 | gloves | 0.102941 | 0.588235 | 136 |
| val low_light_s3 | goggles | 0.021277 | 0.553191 | 47 |
| val low_light_s3 | helmet | 0.054726 | 0.741294 | 201 |
| val low_light_s3 | no_boots | 0.000000 | 0.000000 | 4 |
| val low_light_s3 | no_gloves | 0.107143 | 0.160714 | 56 |
| val low_light_s3 | no_goggle | 0.000000 | 0.024390 | 41 |
| val low_light_s3 | no_helmet | 0.133333 | 0.155556 | 45 |
| val low_light_s3 | none | 0.345679 | 0.481481 | 81 |
| val low_light_s3 | person | 0.581590 | 0.849372 | 239 |
| val low_light_s3 | vest | 0.257310 | 0.754386 | 171 |
| val low_light_s3 | violation_pooled | 0.082192 | 0.116438 | 146 |
- val clean all monitor_alarm_threshold: 0.223440 (n=200)
- train stream all retrain_trigger_window: 7 (n=1)
- train stream all frames_pooled: 400 (n=400)
- train stream all frames_mined: 200 (n=400)
- train stream all replay_frames: 200 (n=332)
- train stream all pseudo_label_error_rate_vs_source: 0.056884 (n=1547)
- train stream all pseudo_label_miss_rate_vs_source: 0.056884 (n=1547)
- train stream violation pseudo_label_error_rate_vs_source: nan (n=0)
- train stream violation pseudo_label_miss_rate_vs_source: 1.000000 (n=88)
- train stream all poisoned_boxes: 88 (n=1547)
- train stream all human_minutes: 0 (n=20)
- train stream all finetune_seconds: 1425.867132 (n=400)
- golden_v1 clean all map50_incumbent: 0.570157 (n=141)
- golden_v1 clean all map50_95_incumbent: 0.290262 (n=141)
- val low_light_s3 all map50_incumbent: 0.253518 (n=143)
- val low_light_s3 all map50_95_incumbent: 0.108700 (n=143)
- golden_v1 low_light_s3 all map50_incumbent: 0.258720 (n=141)
- golden_v1 low_light_s3 all map50_95_incumbent: 0.113625 (n=141)
- golden_v1 clean all map50_candidate: 0.552071 (n=141)
- golden_v1 clean all map50_95_candidate: 0.278591 (n=141)
- val low_light_s3 all map50_candidate: 0.502731 (n=143)
- val low_light_s3 all map50_95_candidate: 0.240725 (n=143)
- golden_v1 low_light_s3 all map50_candidate: 0.474664 (n=141)
- golden_v1 low_light_s3 all map50_95_candidate: 0.227500 (n=141)
- golden_v1 clean all promoted: 0.000000 (n=1)
- train stream all improve_seconds: 1935.577653 (n=400)
- train stream all retrains: 1 (n=25)
