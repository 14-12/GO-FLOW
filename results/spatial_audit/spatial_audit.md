# Audit spatial local — GoFlow / PEMS08

Périmètre : tous les fichiers du projet, y compris les fichiers cachés. Les notebooks sont lus comme documents JSON et ne sont jamais exécutés. Aucun entraînement ni aucune construction de matrice d’adjacence.

## Arborescence complète

```text
GoFlow/
  data/
    raw/
      traffic_dataset.csv
  notebooks/
    01_data_understanding.ipynb
    02_feature_engineering.ipynb
    03_model_benchmark copy.ipynb
    03_model_benchmark.ipynb
  results/
    model_benchmark/
      benchmark_results.csv
      benchmark_results.md
      benchmark_results.xlsx
      lstm_best.pt
      lstm_learning_curve.png
      lstm_normalisation.csv
      lstm_training_history.csv
      patchtst_best.pt
      patchtst_training_history.csv
      xgboost_actual_vs_predicted.png
      xgboost_error_distribution.png
      xgboost_errors_by_occupancy.png
      xgboost_feature_importance.csv
      xgboost_feature_importance.png
      xgboost_validation_results.csv
  scripts/
    audit_spatial_data.py
    run_patchtst_section.py
```

## Inventaire des formats

| Format | Nombre |
| --- | ---: |
| .csv | 7 |
| .ipynb | 4 |
| .md | 1 |
| .png | 5 |
| .pt | 2 |
| .py | 2 |
| .xlsx | 1 |
| .npz | 0 |
| .npy | 0 |
| .pkl | 0 |
| .pickle | 0 |
| .txt | 0 |
| .json | 0 |

## Contenu des fichiers tabulaires

### `data/raw/traffic_dataset.csv`

Chemin absolu : `C:\Users\User\GoFlow\data\raw\traffic_dataset.csv`. Format : CSV. Dimensions : **3,035,520 × 5**.

Mesures temporelles de trafic ; aucune colonne spatiale ni identifiant routier externe.

Colonnes : `timestep`, `location`, `flow`, `occupy`, `speed`.

Premières lignes (valeurs du fichier) :

```csv
timestep,location,flow,occupy,speed
1,0,133.0,0.0603,65.8
1,1,210.0,0.0589,69.6
1,2,124.0,0.0358,65.8
1,3,145.0,0.0416,69.6
1,4,206.0,0.0493,69.4
```

### `results/model_benchmark/benchmark_results.csv`

Chemin absolu : `C:\Users\User\GoFlow\results\model_benchmark\benchmark_results.csv`. Format : CSV. Dimensions : **3 × 24**.

Scores et temps des modèles ; ne représente pas les relations entre capteurs.

Colonnes : `model`, `target`, `horizon_minutes`, `MAE`, `RMSE`, `R2`, `training_time_s`, `tuning_time_s`, `inference_time_s`, `inference_us_per_row`, `validation_RMSE`, `n_train`, `n_validation`, `n_test`, `n_features`, `best_n_estimators`, `xgboost_version`, `n_jobs`, `hyperparameters`, `sequence_length`, `best_epoch`, `epochs_trained`, `torch_version`, `device`.

Premières lignes (valeurs du fichier) :

```csv
model,target,horizon_minutes,MAE,RMSE,R2,training_time_s,tuning_time_s,inference_time_s,inference_us_per_row,validation_RMSE,n_train,n_validation,n_test,n_features,best_n_estimators,xgboost_version,n_jobs,hyperparameters,sequence_length,best_epoch,epochs_trained,torch_version,device
XGBoost Regressor,target_occupy_t15,15,0.0064944570664503,0.0164401864382711,0.8907645943146522,30.325993899954483,101.416394200176,0.4653777999337762,1.0226060777731356,0.0151586628954862,2122450,454410,455090,32,133.0,3.2.0,4,"{""learning_rate"": 0.1, ""max_depth"": 6, ""min_child_weight"": 10, ""n_estimators"": 300}",,,,,
LSTM,target_occupy_t15,15,0.0070899798830753,0.0166211756054034,0.8883462213883311,174.49935820000246,174.49935820000246,0.7641356999520212,1.6790869936760229,0.0153097981736419,2122450,454410,455090,3,,,4,"{""batch_size"": 2048, ""hidden_size"": 16, ""learning_rate"": 0.001, ""max_epochs"": 15, ""num_layers"": 1, ""patience"": 3}",12.0,9.0,12.0,2.11.0+cpu,cpu
PatchTST,target_occupy_t15,15,0.0071745724654265,0.016645690141293,0.8880166226728915,2961.9183225000743,2961.9183225000743,8.772762099979445,19.276982794566887,0.0153128694952306,2120580,454410,455090,3,,,4,"{""batch_size"": 1024, ""d_model"": 64, ""dim_feedforward"": 128, ""dropout"": 0.1, ""learning_rate"": 0.001, ""max_epochs"": 20, ""nhead"": 4, ""num_layers"": 2, ""patch_length"": 4, ""patience"": 4, ""stride"": 4, ""variant"": ""joint_multivariate_patches"", ""weight_decay"": 0.01}",24.0,7.0,11.0,2.11.0+cpu,cpu
```

### `results/model_benchmark/lstm_normalisation.csv`

Chemin absolu : `C:\Users\User\GoFlow\results\model_benchmark\lstm_normalisation.csv`. Format : CSV. Dimensions : **3 × 3**.

Moyennes et écarts-types des trois variables sur train ; aucune relation spatiale.

Colonnes : `feature`, `moyenne_train`, `ecart_type_train`.

Premières lignes (valeurs du fichier) :

```csv
feature,moyenne_train,ecart_type_train
flow,230.33579495394477,145.87929026575762
speed,63.77977403472399,6.560733443470674
occupy,0.0646366002497114,0.0449490071945003
```

### `results/model_benchmark/lstm_training_history.csv`

Chemin absolu : `C:\Users\User\GoFlow\results\model_benchmark\lstm_training_history.csv`. Format : CSV. Dimensions : **12 × 4**.

Historique des époques LSTM ; aucune relation spatiale.

Colonnes : `epoch`, `train_RMSE`, `validation_RMSE`, `epoch_time_s`.

Premières lignes (valeurs du fichier) :

```csv
epoch,train_RMSE,validation_RMSE,epoch_time_s
1,0.0191888271847751,0.0158819372321501,14.863576899981126
2,0.0152135455025448,0.0156345366713764,14.284387800144032
3,0.0151255040990029,0.0155086270340985,14.109197200043129
4,0.0150733655779468,0.0154314957106414,14.299744799965993
5,0.0150300118972984,0.0154250881051817,14.17039469978772
```

### `results/model_benchmark/patchtst_training_history.csv`

Chemin absolu : `C:\Users\User\GoFlow\results\model_benchmark\patchtst_training_history.csv`. Format : CSV. Dimensions : **11 × 3**.

Historique des époques PatchTST ; aucune relation spatiale.

Colonnes : `epoch`, `train_RMSE`, `validation_RMSE`.

Premières lignes (valeurs du fichier) :

```csv
epoch,train_RMSE,validation_RMSE
1,0.0161046575347226,0.0156669522579468
2,0.0151827004985966,0.0156680043910952
3,0.0150624857531155,0.0156144776596394
4,0.01502093505241,0.0155138253356277
5,0.0149513832700051,0.0153431099997524
```

### `results/model_benchmark/xgboost_feature_importance.csv`

Chemin absolu : `C:\Users\User\GoFlow\results\model_benchmark\xgboost_feature_importance.csv`. Format : CSV. Dimensions : **32 × 3**.

Importance des features XGBoost ; location est une feature et non une arête.

Colonnes : `feature`, `gain`, `importance`.

Premières lignes (valeurs du fichier) :

```csv
feature,gain,importance
occupy,16.879226684570312,0.4536094766016916
occupy_rolling_mean_3,12.74908447265625,0.3426167349289066
occupy_lag_1,4.485931396484375,0.1205541599065368
occupy_rolling_mean_6,0.7470345497131348,0.0200756798537765
occupy_rolling_mean_12,0.2622471451759338,0.007047585324053
```

### `results/model_benchmark/xgboost_validation_results.csv`

Chemin absolu : `C:\Users\User\GoFlow\results\model_benchmark\xgboost_validation_results.csv`. Format : CSV. Dimensions : **3 × 10**.

Configurations XGBoost et scores de validation ; aucune relation spatiale.

Colonnes : `essai`, `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `best_n_estimators`, `validation_MAE`, `validation_RMSE`, `validation_R2`, `training_time_s`.

Premières lignes (valeurs du fichier) :

```csv
essai,n_estimators,max_depth,learning_rate,min_child_weight,best_n_estimators,validation_MAE,validation_RMSE,validation_R2,training_time_s
3,300,6,0.1,10,133,0.0060409010047081,0.0151586628954862,0.8926401131074527,30.325993899954483
1,200,4,0.1,5,200,0.0061637816584983,0.0151915699885436,0.8921734840251399,31.76048730011098
2,300,6,0.05,5,167,0.0060371577189406,0.0152041112211922,0.8919953805634979,37.58552870014682
```

## Correspondance des 170 locations

- Nombre de locations : **170**.
- Identifiants locaux observés : `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169]`.
- Timesteps distincts : **17856**, de **1** à **17856**.
- Lignes par location : minimum **17856**, maximum **17856**.
- Couples location/timestep dupliqués : **0**.
- Les valeurs 0 à 169 sont des identifiants locaux. Leur origine et leur correspondance avec des IDs physiques ne sont pas documentées dans les fichiers présents.
- Aucun fichier spatial indépendant ne fournit des IDs à comparer : le mapping est **invérifiable à ce stade**.
- Des IDs consécutifs ne constituent pas une preuve de voisinage. Aucune arête n’est déduite de location.

## Vérification des autres conteneurs

### `results/model_benchmark/benchmark_results.xlsx`

Format : Excel ; tableaux de benchmark.

Feuille `Benchmark` : **3 × 9**.

```csv
Rank,Model,Target,Horizon (min),MAE,RMSE,R²,Training Time (s),Inference Time (s)
1,XGBoost Regressor,target_occupy_t15,15,0.0064944570664503,0.0164401864382711,0.8907645943146522,30.32599389995448,0.4653777999337762
2,LSTM,target_occupy_t15,15,0.0070899798830753,0.0166211756054034,0.8883462213883311,174.4993582000025,0.7641356999520212
3,PatchTST,target_occupy_t15,15,0.007174572465426502,0.01664569014129301,0.8880166226728915,2961.918322500074,8.772762099979445
```

Feuille `Résultats complets` : **3 × 24**.

```csv
model,target,horizon_minutes,MAE,RMSE,R2,training_time_s,tuning_time_s,inference_time_s,inference_us_per_row,validation_RMSE,n_train,n_validation,n_test,n_features,best_n_estimators,xgboost_version,n_jobs,hyperparameters,sequence_length,best_epoch,epochs_trained,torch_version,device
XGBoost Regressor,target_occupy_t15,15,0.0064944570664503,0.0164401864382711,0.8907645943146522,30.32599389995448,101.416394200176,0.4653777999337762,1.022606077773136,0.0151586628954862,2122450,454410,455090,32,133.0,3.2.0,4,"{""learning_rate"": 0.1, ""max_depth"": 6, ""min_child_weight"": 10, ""n_estimators"": 300}",,,,,
LSTM,target_occupy_t15,15,0.0070899798830753,0.0166211756054034,0.8883462213883311,174.4993582000025,174.4993582000025,0.7641356999520212,1.679086993676023,0.0153097981736419,2122450,454410,455090,3,,,4,"{""batch_size"": 2048, ""hidden_size"": 16, ""learning_rate"": 0.001, ""max_epochs"": 15, ""num_layers"": 1, ""patience"": 3}",12.0,9.0,12.0,2.11.0+cpu,cpu
PatchTST,target_occupy_t15,15,0.007174572465426502,0.01664569014129301,0.8880166226728915,2961.918322500074,2961.918322500074,8.772762099979445,19.27698279456689,0.01531286949523066,2120580,454410,455090,3,,,4,"{""batch_size"": 1024, ""d_model"": 64, ""dim_feedforward"": 128, ""dropout"": 0.1, ""learning_rate"": 0.001, ""max_epochs"": 20, ""nhead"": 4, ""num_layers"": 2, ""patch_length"": 4, ""patience"": 4, ""stride"": 4, ""variant"": ""joint_multivariate_patches"", ""weight_decay"": 0.01}",24.0,7.0,11.0,2.11.0+cpu,cpu
```

### `results/model_benchmark/lstm_best.pt`

Format : checkpoint PyTorch. Contenu : poids du modèle et paramètres de prétraitement.

Clés : `['state_dict', 'feature_names', 'sequence_length', 'hidden_size', 'feature_mean', 'feature_scale', 'target_mean', 'target_scale', 'best_epoch']`.
Variables : `['flow', 'speed', 'occupy']` ; longueur de séquence : `12`.
Nombre de tenseurs de poids : **6**. Premières dimensions :

```json
{
  "lstm.weight_ih_l0": [
    64,
    3
  ],
  "lstm.weight_hh_l0": [
    64,
    16
  ],
  "lstm.bias_ih_l0": [
    64
  ],
  "lstm.bias_hh_l0": [
    64
  ],
  "output.weight": [
    1,
    16
  ],
  "output.bias": [
    1
  ]
}
```

Aucune matrice d’adjacence, distance, liste de capteurs physiques ou coordonnées dans ce checkpoint.

### `results/model_benchmark/patchtst_best.pt`

Format : checkpoint PyTorch. Contenu : poids du modèle et paramètres de prétraitement.

Clés : `['state_dict', 'feature_names', 'sequence_length', 'hyperparameters', 'feature_mean', 'feature_scale', 'target_mean', 'target_scale', 'best_epoch']`.
Variables : `['flow', 'speed', 'occupy']` ; longueur de séquence : `24`.
Nombre de tenseurs de poids : **29**. Premières dimensions :

```json
{
  "position": [
    1,
    6,
    64
  ],
  "projection.weight": [
    64,
    12
  ],
  "projection.bias": [
    64
  ],
  "encoder.layers.0.self_attn.in_proj_weight": [
    192,
    64
  ],
  "encoder.layers.0.self_attn.in_proj_bias": [
    192
  ],
  "encoder.layers.0.self_attn.out_proj.weight": [
    64,
    64
  ]
}
```

Aucune matrice d’adjacence, distance, liste de capteurs physiques ou coordonnées dans ce checkpoint.

## Candidats spatiaux et provenance

Fichiers dont le nom indique un contenu spatial : **0**. []
Références spatiales explicites dans les sources des notebooks : **0**. []
Le notebook 01 charge directement traffic_dataset.csv ; aucun code local de conversion d’un NPZ vers le CSV ne documente l’ordre original des capteurs.

Les PNG sont des figures de benchmark, le Markdown est un tableau de scores et le script préexistant exécute la section PatchTST. Ils ne fournissent pas de graphe routier.

## Conclusion : C — aucune donnée spatiale trouvée

Dans les fichiers locaux audités, aucune adjacence 170 × 170, matrice de distance, liste d’arêtes, coordonnée ou liste d’IDs physiques n’a été trouvée. Il n’y a donc aucun fichier candidat spatial dont les dimensions ou premières valeurs puissent être présentées. Le CSV suffit pour décrire les séries temporelles, mais ne permet pas de reconstruire le graphe physique.

## Éléments exacts à récupérer avant STGCN

1. **Le graphe associé à cette version de PEMS08** : soit une matrice d’adjacence **170 × 170**, soit une liste d’arêtes avec `source_sensor_id`, `target_sensor_id` et, si disponibles, `distance` ou `weight`.
2. **La liste ordonnée des 170 capteurs et leur mapping** : table `location,sensor_id`, ou preuve documentée que location correspond à la position du capteur dans le tenseur source et dans les lignes/colonnes du graphe. La seule égalité du nombre de capteurs ne suffit pas.
3. **La provenance de la conversion** : le tenseur PEMS08 original et le code ou manifeste ayant produit traffic_dataset.csv, pour vérifier l’ordre des capteurs et les mesures correspondantes.
4. **La signification du graphe** : arêtes dirigées ou non, unité et définition des distances, sens de circulation, convention des poids et traitement des capteurs sans voisin. Pour une matrice, documenter aussi l’ordre des lignes et colonnes et le sens de la valeur zéro.
5. Si seul un fichier de coordonnées est fourni : récupérer les **IDs physiques, coordonnées et connectivité du réseau routier** (ou un graphe dérivé documenté). Les coordonnées seules ne prouvent pas la connexion routière entre deux capteurs.

Ne créer aucune matrice d’adjacence avant d’avoir vérifié cette correspondance. Aucun fichier n’est téléchargé dans cet audit.

## Vérification de conservation

Empreintes SHA-256 identiques avant et après lecture pour tous les fichiers inventoriés : notebooks, checkpoints et résultats conservés. Aucun modèle exécuté.
