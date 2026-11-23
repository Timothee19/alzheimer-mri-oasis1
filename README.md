# Détection de la maladie d'Alzheimer par IRM (OASIS-1)

Projet d'analyse exploratoire (EDA) et de modélisation Deep Learning sur le jeu de données d'IRM cérébrales **OASIS-1** (**86 437 coupes 2D**, **1 417 volumes MPR**, **347 patients uniques**), exécuté sous TensorFlow avec accélération GPU **Intel Iris XPU (ITEX)**.

---

## Organisation des Notebooks (v1 & v2)

Les deux séries d'expériences (**v1 originale** et **v2 optimisée**) sont conservées côte à côte dans le dépôt afin de permettre la comparaison directe des choix d'architecture, de régularisation et de suivi d'entraînement.

### Analyse exploratoire (commune)
- **[`EDA_Alzheimer_OASIS1.ipynb`](EDA_Alzheimer_OASIS1.ipynb)** — Analyse complète du dataset :
  - Hiérarchie *Patient → Session (`MR`) → Volume (`mpr`) → Coupe (`100..160`)* (61 coupes/volume).
  - Déséquilibre des classes (images vs patients, dont seulement 2 patients en `Moderate Dementia`) et mise en évidence du risque de *data leakage* si le split n'est pas fait par patient.
  - Intégrité des fichiers (0 corrompu, 0 doublon MD5, dimensions uniformes `496×248`, canaux RGB identiques).
  - Statistiques d'intensité, cartes moyennes/écarts-types et analyse PCA.

### Version 1 — Première itération (originale)
1. **[`01_Split_et_Baseline_2D.ipynb`](01_Split_et_Baseline_2D.ipynb)** — Split strict par patient (`subject_id`, 70/15/15 %) sauvegardé dans `artifacts_modeling/splits_binary.csv`, puis baseline **CNN 2D** binaire (`Non Demented` vs `Demented`) coupe par coupe avec évaluation aux niveaux coupe, volume et patient.
2. **[`02_Modele_61_Coupes_MIL.ipynb`](02_Modele_61_Coupes_MIL.ipynb)** — Modèle **61 coupes d'un coup** (`Conv1D + Gated Attention MIL`) sur les embeddings `(61, 64)` → probabilité binaire unique `P(Demented)` + visualisation des poids d'attention sur les coupes `100 → 160` *(monitoring initial sur `val_loss` avec `L2 = 1e-2`)*.
3. **[`03_Transfer_Learning_MobileNetV2_MIL.ipynb`](03_Transfer_Learning_MobileNetV2_MIL.ipynb)** — Transfer Learning **MobileNetV2 (ImageNet)** en entrée **2.5D `(t-1, t, t+1)`** (Phase 1 warm-up + Phase 2 fine-tuning doux des 15 dernières couches à `lr = 1e-5`) suivi du modèle 61 coupes MIL.

### Version 2 — Itération optimisée (`_v2`)
Ce qui change entre **v1** et **v2** :
- **Correction du monitoring des modèles MIL (`02_v2` & `03_v2`)** : en v1, la forte régularisation `L2 = 1e-2` faisait baisser artificiellement `val_loss` jusqu'à l'époque 50 alors que `val_roc_auc` et `val_pr_auc` déclinaient dès les époques 5–15. En v2, `ModelCheckpoint`, `ReduceLROnPlateau` et `EarlyStopping` suivent **`val_roc_auc` (`mode="max"`)** avec `L2 = 1e-3`, et les courbes affichent le repère exact de la meilleure époque restaurée.
- **Amélioration du fine-tuning MobileNetV2 (`03_v2`)** : dégel des **35 dernières couches** (hors `BatchNormalization`) avec `lr = 3e-5`, `STEPS_PER_EPOCH = 120`, `EPOCHS_FINETUNE = 10`, rechargement du meilleur checkpoint Phase 1 et sauvegarde conditionnée à l'amélioration de `val_roc_auc` (`initial_value_threshold`).
- **Alignement documentation / code** (`64 dims`, `160` filtres, noms de caches cohérents).

Notebooks v2 :
1. **[`01_Split_et_Baseline_2D_v2.ipynb`](01_Split_et_Baseline_2D_v2.ipynb)**
2. **[`02_Modele_61_Coupes_MIL_v2.ipynb`](02_Modele_61_Coupes_MIL_v2.ipynb)**
3. **[`03_Transfer_Learning_MobileNetV2_MIL_v2.ipynb`](03_Transfer_Learning_MobileNetV2_MIL_v2.ipynb)**

---

## Protocole d'évaluation (sans *data leakage*)

Le split est strictement stratifié au niveau des **347 patients uniques** (0 patient partagé entre Train, Val et Test) et identique pour tous les notebooks :

| Split | Patients (Sains / Malades) | Volumes MPR | Coupes 2D |
| :--- | :---: | :---: | :---: |
| **Train (70 %)** | 242 (186 / 56) | 984 | 60 024 |
| **Validation (15 %)** | 52 (40 / 12) | 213 | 12 993 |
| **Test (15 %)** | 53 (40 / 13 : 9 *Very mild*, 4 *Mild*) | 220 | 13 420 |

Le seuil de décision binaire est calibré uniquement sur le jeu de **Validation** (indice de Youden au niveau patient), puis évalué sans biais sur le jeu de **Test**.

---

## Résultats comparatifs (Test Set — 53 patients inédits)

### Version 1 (originale)

| Modèle (Test — niveau Patient) | ROC-AUC | PR-AUC | Balanced Acc | F1 (Malade) | Spécificité / Rappel |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `01` — Baseline CNN 2D (*from scratch*, moy. coupes) | **0.9115** | 0.6687 | 0.8250 | 0.6500 | 65,0 % / **100 %** (13/13) |
| `02` — CNN 2D + MIL 61 coupes | 0.8923 | **0.6870** | 0.8240 | 0.6667 | **72,5 %** / 92,3 % (12/13) |
| `03a` — MobileNetV2 2.5D (moy. coupes) | 0.8615 | 0.5935 | **0.8625** | **0.7027** | — |
| `03b` — MobileNetV2 + MIL 61 coupes | 0.8442 | 0.5575 | 0.8115 | 0.6486 | 70,0 % / 92,3 % (12/13) |

### Version 2 (`_v2`, optimisée)

| Modèle (Test — niveau Patient) | ROC-AUC | PR-AUC | Balanced Acc | F1 (Malade) | Spécificité / Rappel |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `01_v2` — Baseline CNN 2D (*from scratch*, moy. coupes) | 0.8442 | 0.5199 | 0.7721 | 0.6250 | 77,5 % / 76,9 % (10/13) |
| `02_v2` — CNN 2D + MIL 61 coupes | 0.8731 | 0.6055 | **0.8625** | 0.7027 | 72,5 % / **100 %** (13/13) |
| `03a_v2` — MobileNetV2 2.5D (moy. coupes) | **0.8827** | **0.6288** | 0.8615 | **0.7273** | — |
| `03b_v2` — MobileNetV2 + MIL 61 coupes | 0.8808 | 0.6189 | 0.7971 | 0.6667 | **82,5 %** / 76,9 % (Acc **81,1 %**) |

---

## Analyse technique & enseignements clés

1. **Pourquoi autant de `Non-trainable params` dans les résumés ?**
   - Dans les CNN et MIL (`01`, `02`), les quelques paramètres non entraînables (`1 536` et `48`) correspondent aux statistiques mobiles (`moving_mean`, `moving_variance`) des couches `BatchNormalization`.
   - Dans `03` (**MobileNetV2**), `model.summary()` affiche **2 257 984 paramètres non entraînables (96,5 %)** car il est appelé en **Phase 1 (warm-up)** où tout le backbone ImageNet est volontairement gelé pour entraîner uniquement la tête (`82 049` paramètres). En **Phase 2 (`03_v2`)**, le dégel des 35 dernières couches (hors `BatchNormalization`) porte le nombre de paramètres entraînables à **1 690 113**.
2. **Impact de la correction MIL (`v1 → v2`)** :
   - Sur `02_v2`, le passage au monitoring `val_roc_auc` élimine totalement le sur-apprentissage (arrêt propre à l'époque 11, restauration de la meilleure époque) et fait dépasser la Baseline 2D sur **toutes** les métriques (`+2,9 pts` ROC-AUC, `+8,6 pts` PR-AUC, `+9,0 pts` Balanced Acc, **100 % de rappel** avec 0 faux négatif).
   - Sur `03_v2`, le fine-tuning étendu (35 couches, `lr = 3e-5`) propulse **MobileNetV2 2.5D** en tête des modèles v2 (**ROC-AUC `0.8827`**, **PR-AUC `0.6288`**, **F1 `0.7273`**) et améliore **MobileNetV2 + MIL** de `+3,7 pts` en ROC-AUC (`0.8442 → 0.8808`) avec la meilleure précision globale (**81,1 %**, seulement 7 faux positifs sur 40 sains).
3. **Lecture des courbes Train vs Val (`03_v2`)** :
   - En Phase 2 de MobileNetV2 (1,69 M de paramètres pour 242 patients en Train), un écart Train/Val apparaît naturellement en fin d'entraînement, mais la validation reste stable et le checkpoint retenu (époque 7) coïncide à la fois avec le pic `val_roc_auc` (`0.8484`) et le minimum de `val_loss` (`0.5167`).
   - Sur le MIL en 2 étapes (`03_v2`), l'écart initial Train/Val dès l'époque 1 provient du fait que les embeddings `(61, 64)` sont extraits par un backbone déjà entraîné sur le Train ; les courbes de validation restent ensuite parfaitement stables sans sur-apprentissage du MIL.
4. **Sensibilité à la variance sur petit échantillon** :
   - Entre `01` (v1) et `01_v2`, le CNN 2D *from scratch* montre une variance notable d'un run à l'autre (`ROC-AUC 0.912` vs `0.844` sur 13 patients malades en Test), alors que **MobileNetV2 2.5D** (pré-entraîné ImageNet) et l'agrégation **MIL 61 coupes** apportent une stabilité nettement supérieure.

---

## Données

Les images IRM brutes (~1,3 Go, 86 437 images) sont hébergées dans la **GitHub Release [`v1.0-data`](https://github.com/Timothee19/alzheimer-mri-oasis1/releases/tag/v1.0-data)** afin de garder le dépôt Git léger et rapide à cloner.

### Téléchargement rapide (après `git clone`)

```bash
# 1. Télécharger et décompresser les 4 dossiers d'IRM à la racine du projet
curl -L -o oasis1_alzheimer_mri_data.zip \
  https://github.com/Timothee19/alzheimer-mri-oasis1/releases/download/v1.0-data/oasis1_alzheimer_mri_data.zip
unzip -q oasis1_alzheimer_mri_data.zip && rm oasis1_alzheimer_mri_data.zip

# 2. (Optionnel) Récupérer le split patient exact utilisé dans les notebooks
mkdir -p artifacts_modeling
curl -L -o artifacts_modeling/splits_binary.csv \
  https://github.com/Timothee19/alzheimer-mri-oasis1/releases/download/v1.0-data/splits_binary.csv
```

Structure attendue après décompression :

```text
.
├── Non Demented/
├── Very mild Dementia/
├── Mild Dementia/
├── Moderate Dementia/
├── artifacts_modeling/
│   └── splits_binary.csv
├── EDA_Alzheimer_OASIS1.ipynb
├── 01_Split_et_Baseline_2D.ipynb
├── 02_Modele_61_Coupes_MIL.ipynb
├── 03_Transfer_Learning_MobileNetV2_MIL.ipynb
├── 01_Split_et_Baseline_2D_v2.ipynb
├── 02_Modele_61_Coupes_MIL_v2.ipynb
├── 03_Transfer_Learning_MobileNetV2_MIL_v2.ipynb
└── README.md
```

---

## Environnement

- **Conda** : `tf-iris-fedora` (Python 3.11, TensorFlow 2.15 + Intel Extension for TensorFlow XPU `/XPU:0`, NumPy, Pandas, Matplotlib, SciPy, scikit-learn, Pillow).
