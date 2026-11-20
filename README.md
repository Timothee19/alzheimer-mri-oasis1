# Détection de la maladie d'Alzheimer par IRM (OASIS-1)

Projet d'analyse exploratoire (EDA) et de modélisation Deep Learning sur le jeu de données d'IRM cérébrales **OASIS-1** (86 437 coupes 2D issues de 347 patients uniques et 1 417 volumes MPR), exécuté sous TensorFlow avec accélération GPU **Intel Iris XPU (ITEX)**.

## Notebooks

1. **[`EDA_Alzheimer_OASIS1.ipynb`](EDA_Alzheimer_OASIS1.ipynb)** — Analyse exploratoire complète :
   - Hiérarchie *Patient → Session (`MR`) → Volume (`mpr`) → Coupe (`100..160`)* (61 coupes/volume).
   - Déséquilibre des classes (images vs patients, dont seulement 2 patients en `Moderate Dementia`) et mise en évidence du risque de *data leakage*.
   - Intégrité des fichiers (0 corrompu, 0 doublon MD5, dimensions uniformes `496×248`, canaux RGB identiques).
   - Statistiques d'intensité, cartes moyennes/écarts-types et analyse PCA.
2. **[`01_Split_et_Baseline_2D.ipynb`](01_Split_et_Baseline_2D.ipynb)** — Split strict par patient (`subject_id`, 70/15/15 %) et baseline **CNN 2D** binaire (`Non Demented` vs `Demented`) coupe par coupe avec évaluation aux niveaux coupe, volume et patient.
3. **[`02_Modele_61_Coupes_MIL.ipynb`](02_Modele_61_Coupes_MIL.ipynb)** — Modèle **61 coupes d'un coup** (`Conv1D + Gated Attention MIL`) sur les embeddings `(61, 128)` → probabilité binaire unique `P(Demented)` et visualisation des poids d'attention sur les coupes `100 → 160`.
4. **[`03_Transfer_Learning_MobileNetV2_MIL.ipynb`](03_Transfer_Learning_MobileNetV2_MIL.ipynb)** — Transfer Learning **MobileNetV2 (ImageNet)** en entrée **2.5D `(t-1, t, t+1)`** (warm-up + fine-tuning) suivi du modèle 61 coupes MIL, comparé directement à la baseline sur le même split patient.

## Données

Les images IRM brutes (~1,4 Go) ne sont pas versionnées dans ce dépôt. Placez les 4 dossiers du dataset OASIS-1 à la racine du projet :

```text
.
├── Non Demented/
├── Very mild Dementia/
├── Mild Dementia/
├── Moderate Dementia/
├── EDA_Alzheimer_OASIS1.ipynb
├── 01_Split_et_Baseline_2D.ipynb
├── 02_Modele_61_Coupes_MIL.ipynb
└── 03_Transfer_Learning_MobileNetV2_MIL.ipynb
```

## Environnement

- Conda : `tf-iris-fedora` (Python 3.11, TensorFlow 2.15 + Intel Extension for TensorFlow XPU, NumPy, Pandas, Matplotlib, SciPy, scikit-learn, Pillow).
