# Clustering: all real dataset/seed rows

Columns group strategy × metric. Empty cells mean unrun/invalid, never a hidden zero.

| Dataset / seed | baseline: ari, ami; seconds; valid; fallback; selected | auto: ari, ami; seconds; valid; fallback; selected | routed: ari, ami; seconds; valid; fallback; selected |
|---|---|---|---|
| iris / 42 | 0.61007, 0.64821; 5.58s; True; none; KMeans | 0.90387, 0.89844; 5.32s; True; none; GaussianMixtureFull | 0.90387, 0.89844; 7.55s; True; none; GaussianMixtureFull |
| iris / 314 | 0.62014, 0.65522; 5.30s; True; none; KMeans | 0.90387, 0.89844; 5.45s; True; none; GaussianMixtureFull | 0.90387, 0.89844; 7.32s; True; none; GaussianMixtureFull |
| iris / 2027 | 0.62014, 0.65522; 5.32s; True; none; KMeans | 0.90387, 0.89844; 5.29s; True; none; GaussianMixtureFull | 0.90387, 0.89844; 7.78s; True; none; GaussianMixtureFull |
| wine / 42 | 0.89749, 0.87458; 5.40s; True; none; KMeans | 0.89749, 0.87458; 5.24s; True; none; KMeans | 0.89749, 0.87458; 8.39s; True; none; KMeans |
| wine / 314 | 0.89749, 0.87458; 5.32s; True; none; KMeans | 0.89749, 0.87458; 5.34s; True; none; KMeans | 0.89749, 0.87458; 7.51s; True; none; KMeans |
| wine / 2027 | 0.86360, 0.84567; 5.40s; True; none; KMeans | 0.86360, 0.84567; 5.31s; True; none; KMeans | 0.86360, 0.84567; 7.57s; True; none; KMeans |
| digits / 42 | 0.56014, 0.69371; 5.41s; True; none; KMeans | 0.56014, 0.69371; 5.60s; True; none; KMeans | 0.56014, 0.69371; 7.92s; True; none; KMeans |
| digits / 314 | 0.52990, 0.66174; 5.35s; True; none; KMeans | 0.52990, 0.66174; 5.31s; True; none; KMeans | 0.52990, 0.66174; 7.63s; True; none; KMeans |
| digits / 2027 | 0.54197, 0.67081; 5.34s; True; none; KMeans | 0.54197, 0.67081; 5.45s; True; none; KMeans | 0.54197, 0.67081; 7.96s; True; none; KMeans |
