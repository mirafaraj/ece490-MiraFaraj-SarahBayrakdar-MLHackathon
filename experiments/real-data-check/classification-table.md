# Classification: all real dataset/seed rows

Columns group strategy × metric. Empty cells mean unrun/invalid, never a hidden zero.

| Dataset / seed | baseline: accuracy, balanced_accuracy, macro_f1, minority_recall; seconds; valid; fallback; selected | auto: accuracy, balanced_accuracy, macro_f1, minority_recall; seconds; valid; fallback; selected | routed: accuracy, balanced_accuracy, macro_f1, minority_recall; seconds; valid; fallback; selected |
|---|---|---|---|
| breast_cancer / 42 | 0.95804, 0.95115, 0.95467, 0.92453; 5.83s; True; none; ExtraTreesClassifier | 0.96503, 0.96834, 0.96294, 0.98113; 9.29s; True; none; linear | 0.96503, 0.96834, 0.96294, 0.98113; 11.11s; True; none; linear |
| breast_cancer / 314 | 0.96503, 0.96447, 0.96267, 0.96226; 5.53s; True; none; ExtraTreesClassifier | 0.96503, 0.96447, 0.96267, 0.96226; 8.79s; True; none; extra_trees | 0.96503, 0.96447, 0.96267, 0.96226; 10.58s; True; none; extra_trees |
| breast_cancer / 2027 | 0.99301, 0.99057, 0.99248, 0.98113; 5.40s; True; none; ExtraTreesClassifier | 0.99301, 0.99057, 0.99248, 0.98113; 10.32s; True; none; histogram | 0.99301, 0.99057, 0.99248, 0.98113; 13.03s; True; none; histogram |
| wine / 42 | 1.00000, 1.00000, 1.00000, 1.00000; 5.21s; True; none; ExtraTreesClassifier | 1.00000, 1.00000, 1.00000, 1.00000; 5.52s; True; none; extra_trees | 1.00000, 1.00000, 1.00000, 1.00000; 7.54s; True; none; extra_trees |
| wine / 314 | 1.00000, 1.00000, 1.00000, 1.00000; 5.94s; True; none; ExtraTreesClassifier | 1.00000, 1.00000, 1.00000, 1.00000; 6.08s; True; none; extra_trees | 1.00000, 1.00000, 1.00000, 1.00000; 8.39s; True; none; extra_trees |
| wine / 2027 | 0.95556, 0.95370, 0.95370, 0.91667; 5.18s; True; none; ExtraTreesClassifier | 0.95556, 0.96296, 0.95475, 1.00000; 5.79s; True; none; linear | 0.95556, 0.96296, 0.95475, 1.00000; 8.01s; True; none; linear |
| digits / 42 | 0.98222, 0.98186, 0.98198, 0.90698; 6.03s; True; none; ExtraTreesClassifier | 0.96667, 0.96634, 0.96628, 0.88372; 10.18s; True; none; histogram | 0.96667, 0.96634, 0.96628, 0.88372; 12.94s; True; none; histogram |
| digits / 314 | 0.98444, 0.98424, 0.98424, 0.95349; 6.17s; True; none; ExtraTreesClassifier | 0.97778, 0.97767, 0.97760, 0.95349; 9.82s; True; none; histogram | 0.97778, 0.97767, 0.97760, 0.95349; 13.17s; True; none; histogram |
| digits / 2027 | 0.97333, 0.97291, 0.97285, 0.88372; 6.04s; True; none; ExtraTreesClassifier | 0.95778, 0.95714, 0.95756, 0.81395; 7.33s; True; none; linear | 0.95778, 0.95714, 0.95756, 0.81395; 10.68s; True; none; linear |
| iris / 42 | 0.92105, 0.92308, 0.92296, 0.88462; 5.65s; True; none; ExtraTreesClassifier | 0.92105, 0.92308, 0.92296, 0.88462; 5.64s; True; none; linear | 0.92105, 0.92308, 0.92296, 0.88462; 8.16s; True; none; linear |
| iris / 314 | 0.92105, 0.92094, 0.92000, 0.92308; 5.34s; True; none; ExtraTreesClassifier | 0.92105, 0.92094, 0.92000, 0.92308; 6.12s; True; none; extra_trees | 0.92105, 0.92094, 0.92000, 0.92308; 8.73s; True; none; extra_trees |
| iris / 2027 | 0.92105, 0.92308, 0.92296, 0.88462; 5.50s; True; none; ExtraTreesClassifier | 0.92105, 0.92308, 0.92296, 0.88462; 5.61s; True; none; extra_trees | 0.92105, 0.92308, 0.92296, 0.88462; 7.72s; True; none; extra_trees |
| mammography / 42 | 0.98677, 0.76044, 0.82044, 0.52308; 6.87s; True; none; ExtraTreesClassifier | 0.90451, 0.88353, 0.62215, 0.86154; 7.65s; True; none; linear | 0.90451, 0.88353, 0.62215, 0.86154; 13.05s; True; none; linear |
| mammography / 314 | 0.98534, 0.72967, 0.79330, 0.46154; 7.74s; True; none; ExtraTreesClassifier | 0.90021, 0.86632, 0.61273, 0.83077; 7.14s; True; none; linear | 0.90021, 0.86632, 0.61273, 0.83077; 12.34s; True; none; linear |
| mammography / 2027 | 0.98462, 0.72930, 0.78735, 0.46154; 8.13s; True; none; ExtraTreesClassifier | 0.97532, 0.85220, 0.78199, 0.72308; 9.22s; True; none; histogram | 0.97532, 0.85220, 0.78199, 0.72308; 15.26s; True; none; histogram |
| blood_transfusion / 42 | 0.68984, 0.60052, 0.59360, 0.43182; 6.24s; True; none; ExtraTreesClassifier | 0.69519, 0.69843, 0.64874, 0.70455; 6.00s; True; none; histogram | 0.69519, 0.69843, 0.64874, 0.70455; 9.64s; True; none; histogram |
| blood_transfusion / 314 | 0.67914, 0.55420, 0.55420, 0.31818; 5.71s; True; none; ExtraTreesClassifier | 0.64171, 0.67133, 0.60641, 0.72727; 5.67s; True; none; linear | 0.64171, 0.67133, 0.60641, 0.72727; 7.99s; True; none; linear |
| blood_transfusion / 2027 | 0.70053, 0.58271, 0.58392, 0.35556; 6.13s; True; none; ExtraTreesClassifier | 0.65775, 0.69875, 0.62786, 0.77778; 5.54s; True; none; linear | 0.65775, 0.69875, 0.62786, 0.77778; 8.43s; True; none; linear |
