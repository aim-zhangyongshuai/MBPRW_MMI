# README

Code used for the analysis of the paper:

**A multimodal network medicine framework explains general therapeutical associations and enables mechanism-constrained therapeutical predictions**

*Yongshuai Zhang1†, Zhenghao Li1,2†, Jiating Yu3, Jun Xu1, Xiao Gan1**

---

## Data Overview

### 1. Treatment Entity–Target and Disease–Protein Association Data (`data/drug_disease/`)

#### Chemical–Disease
- `CTD_chem_gene_ixns_filtered_renew.csv` – Chemical–target protein associations.  
  Columns: `ChemicalID`, `GeneSymbol`.
- `CTD_genes_diseases_association_genes_20.csv` – Disease–protein associations.  
  Columns: `DiseaseID`, `GeneSymbol`.

#### Drug–Disease
- `1_drug_to_protein_update.csv` – Drug–target protein associations.  
  Columns: `node_1` (drug ID), `node_2_name` (gene symbol).
- `2_disease_to_protein_association_genes_20.csv` – Disease–protein associations.  
  Columns: `node_1` (disease ID), `node_2_name` (gene symbol).

#### Herb–Symptom
- `S4-1.HIT_herb_target_data_0412dropna_filter_ncRNA.csv` – Herb–target protein associations.  
  Columns: `tcm_id`, `Gene Symbol`.
- `S1.TCM_symptom_genes_association_genes_20.csv` – Symptom–protein associations.  
  Columns: `Symptom`, `Symbol`.

### 2. Known Treatment Associations (Ground Truth) (`data/ground_truth/`)
- `CTD_chemicals_diseases_filtered.csv` – Valid Chemical–Disease associations (11,106 pairs).  
  Columns: `ChemicalID`, `DiseaseID`.
- `6_drug_indication_df.csv` – Valid Drug–Disease associations (3,149 pairs).  
  Columns: `drug_id`, `disease_id`.
- `indication_result_TCM.csv` – Valid Herb–Symptom associations (1,436 pairs).  
  Columns: `herb_id`, `symptom_id`.

### 3. Network Edge Data (for building the MMI network) (`data/network_edge/`)
- `Protein_to_GO_formal_version.csv` – Protein–GO interactions.
- `GO_to_GO_filtered.csv` – GO–GO interactions.
- `ncPPI_PPI_filter_RNA_Icc.csv` – Protein–Protein interactions.

### 4. Random Walk (MBPRW) Data (`data/Random walk network data/`)

#### Network node type files
- `node_types_CTD.csv` – Node types for Chemical–Disease task (`drug`, `disease`, `gene`, `GO`).
- `node_types_drug.csv` – Node types for Drug–Disease task (`drug`, `disease`, `gene`, `GO`).
- `node_types_TCM_SID.csv` – Node types for Herb–Symptom task (`drug`, `disease`, `gene`, `GO`).

#### Network graph data (MMI edge lists augmented with treatment entity and disease nodes)
- `processed_graph_add_CTD.csv` – MMI edge list for Chemical–Disease task.
- `processed_graph_add_drug.csv` – MMI edge list for Drug–Disease task.
- `processed_graph_add_TCM.csv` – MMI edge list for Herb–Symptom task.

#### Labeled treatment–disease pair files (for XGBoost training/testing)
- `dda_CTD.csv` – Positive and negative sample pairs for Chemical–Disease.
- `dda_drug.csv` – Positive and negative sample pairs for Drug–Disease.
- `dda_TCM.csv` – Positive and negative sample pairs for Herb–Symptom.

### 5. Clinical Validation Dataset (`data/Clinical Label Dataset/`)
- `clinical occurrence_pairs.tsv` – Clinical occurrence associations (4,869 pairs).
- `clinical effectiveness_pairs.tsv` – Clinical effectiveness associations (760 pairs).
- `clinical PSM effectiveness_pairs.tsv` – Associations significant after propensity score matching (65 pairs).

---

## Scripts Description

### 1. `save_matrix.py`
- **Function**: Precompute the weighted shortest path matrix between all protein nodes in the MMI network for given λ values.
- **Input**: PPI edge file, protein-GO interaction file, GO-GO interaction file; list of λ values.
- **Output**: `adj_matrix_1_{λ}_{λ}.pkl` (distance matrix), `node_to_index_1_{λ}_{λ}.pkl` (index mapping), `graph.pkl` (full network graph).

### 2. `proximity with different λ.py`
- **Function**: Batch calculate the network proximity (z-score) of all treatment-disease pairs under different λ values.
- **Input**: Treatment-target file, disease-protein file, precomputed distance matrix and index mapping, background protein network graph, ground truth label file.
- **Output**: `CTD_proximity_λ_{λ}.csv` (columns: d, z, mean, std). Label merging required in subsequent steps.

### 3. `check_model_performance.py`
- **Function**: Evaluate the prediction performance of network proximity across different λ values (AUC, Precision@Top1%, Recall@Top1%).
- **Input**: Proximity CSV files with the `indication` column (e.g., `CTD_λ_{λ}_with_indication.csv`).
- **Output**: `net_performance_MMI.xlsx` (performance summary for all tasks and λ values).

### 4. `cal_EPS.py`
- **Function**: Calculate the Effective Path Subgraph (EPS) for all treatment-disease pairs (Cartesian product).
- **Input**: Treatment-target file, disease-protein file, network edge file, list of λ values.
- **Output**: Path files named `{TreatmentID}_{DiseaseID}_{λ}.txt` (each records shortest paths), saved under `output/Path_analysis/{CD/DD/HS}_update/lambda_{λ}/`.

### 5. `cal_EPS_ground_truth.py`
- **Function**: Calculate EPS only for positive pairs randomly sampled from the ground truth dataset.
- **Input**: Same as `cal_EPS.py`, plus ground truth file. Sampling parameters: `num_pairs=200`, `seed=678`.
- **Output**: Path files under `output/Path_analysis/{CD/DD/HS}_ground_truth/lambda_{λ}/`, and index file `pairs_to_index_drug.csv`.

### 6. `r_network_change.py`
- **Function**: Compute the change ratio `r(λ)` of shortest path matrices for each λ, with λ=5 as the baseline.
- **Input**: `adj_matrix_1_{λ}_{λ}.pkl` for all λ, baseline matrix `adj_matrix_1_5_5.pkl`; optional: `graph_PPI.pkl` and any `node_to_index_*.pkl`.
- **Output**: `r_network_change.csv` (columns: λ, r, same_count, diff_count).

### 7. `run.py`
- **Function**: One-click execution of the full MBPRW workflow: Embedding Learning → Model Training → Clinical Prediction.
- **Input**: MMI edge list, node type file, training label file, three clinical association files.
- **Output**: Node embedding file, trained model, prediction results for training & test sets, and prediction scores for three clinical datasets.
