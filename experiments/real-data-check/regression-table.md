# Regression: all real dataset/seed rows

Columns group strategy × metric. Empty cells mean unrun/invalid, never a hidden zero.

| Dataset / seed | baseline: rmse, mae, r2; seconds; valid; fallback; selected | auto: rmse, mae, r2; seconds; valid; fallback; selected | routed: rmse, mae, r2; seconds; valid; fallback; selected |
|---|---|---|---|
| diabetes / 42 | 51.72851, 42.04974, 0.51610; 5.86s; True; none; ExtraTreesRegressor | 53.31824, 41.50734, 0.48590; 5.58s; True; none; linear | 53.31824, 41.50734, 0.48590; 7.57s; True; none; linear |
| diabetes / 314 | 61.69144, 50.19191, 0.38047; 5.41s; True; none; ExtraTreesRegressor | 57.47303, 46.06796, 0.46230; 5.50s; True; none; linear | 57.47303, 46.06796, 0.46230; 8.17s; True; none; linear |
| diabetes / 2027 | 54.22007, 43.70073, 0.48415; 5.14s; True; none; ExtraTreesRegressor | 52.78819, 42.92994, 0.51103; 5.37s; True; none; linear | 52.78819, 42.92994, 0.51103; 7.50s; True; none; linear |
| california_housing / 42 | 0.50759, 0.33111, 0.80529; 11.22s; True; none; ExtraTreesRegressor | 0.48642, 0.33195, 0.82119; 6.31s; True; none; histogram | 0.48642, 0.33195, 0.82119; 14.53s; True; none; histogram |
| california_housing / 314 | 0.50615, 0.33082, 0.80938; 12.05s; True; none; ExtraTreesRegressor | 0.49366, 0.33621, 0.81867; 7.86s; True; none; histogram | 0.49366, 0.33621, 0.81867; 14.17s; True; none; histogram |
| california_housing / 2027 | 0.48535, 0.32275, 0.82492; 12.17s; True; none; ExtraTreesRegressor | 0.47469, 0.33004, 0.83252; 7.55s; True; none; histogram | 0.47469, 0.33004, 0.83252; 15.70s; True; none; histogram |
| house_prices / 42 | 28086.36277, 17899.31160, 0.88739; 25.20s; True; none; ExtraTreesRegressor | 27215.12186, 16231.64705, 0.89427; 15.57s; True; none; histogram | 27215.12186, 16231.64705, 0.89427; 36.53s; True; none; histogram |
| house_prices / 314 | 25189.94949, 17378.34171, 0.89455; 18.55s; True; none; ExtraTreesRegressor | 24561.28737, 16376.21767, 0.89974; 13.97s; True; none; histogram | 24561.28737, 16376.21767, 0.89974; 34.63s; True; none; histogram |
| house_prices / 2027 | 39088.58642, 20975.02104, 0.79512; 17.61s; True; none; ExtraTreesRegressor | 39088.58642, 20975.02104, 0.79512; 23.12s; True; none; extra_trees | 39088.58642, 20975.02104, 0.79512; 25.87s; True; none; extra_trees |
