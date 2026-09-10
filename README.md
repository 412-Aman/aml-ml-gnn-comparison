# Transaction-Level AML Model Comparison

This repository contains the reproducibility materials for the paper titled "Comparative Analysis of Traditional Machine Learning and Graph Neural Network Models for Anti-Money Laundering Detection."

The study compares Random Forest, XGBoost, Linear Support Vector Machine, Graph Convolutional Network, GraphSAGE, and Graph Attention Network on a common transaction-level prediction task.

## Dataset

The experiment uses `HI-Small_Trans.csv` from the IBM Transactions for Anti Money Laundering dataset:

https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

The dataset is not redistributed in this repository. Add the Kaggle dataset to the notebook before running it. The notebook automatically searches `/kaggle/input` for `HI-Small_Trans.csv`.

## Kaggle instructions

1. Create a new Kaggle notebook.
2. Add the IBM Transactions for Anti Money Laundering dataset.
3. Enable a GPU accelerator. The recorded experiment used an NVIDIA Tesla T4.
4. Upload `notebooks/AML_complete_experiment_notebook.ipynb` or import it into Kaggle.
5. Select Run All and keep the session active until every cell finishes.
6. Generated artifacts will be written to `/kaggle/working/aml_outputs`.
7. Download the final ZIP from the Kaggle output directory.

The notebook is checkpointed throughout the workflow. Existing model files are reused when a session is resumed.

## Experimental protocol

- Complete IBM AML HI-Small transaction file
- Nine exact duplicate rows removed
- Chronological split of approximately 60 percent training, 20 percent validation, and 20 percent test
- Identical timestamps kept within one partition
- Training-only preprocessing and imbalance treatment
- No SMOTE
- Validation and test sets retain their natural class distributions
- Common transaction-level target for all six models
- Validation and test edges excluded from GNN message passing
- Five seeds: 42, 52, 62, 72, and 82

## Metric terminology

The files retain the historical field name `auprc`. Its value was calculated with scikit-learn `average_precision_score`. It should therefore be interpreted as Average Precision, a precision-recall summary metric.

## Repository contents

- `notebooks`: ordered Kaggle notebook containing the complete experiment workflow
- `results/tables`: final metrics, confidence intervals, and GNN ablation results
- `results/figures`: manuscript performance and SHAP figures
- `results/explainability`: SHAP and GNNExplainer records
- `audit`: dataset audit, feature configuration, split summary, and training sampling records
- `environment`: recorded software and hardware information
- `docs`: experimental protocol and ablation scope

Only final converged results should be used. Earlier eight-epoch GNN runs were diagnostic pilots and are not the reported results.

## Complete experiment archive

The complete results ZIP contains predictions and additional intermediate outputs. Because it is larger than the normal GitHub file limit, publish it as an asset in the `v1.0.0` GitHub Release rather than committing it to the repository.

## Reproducibility limitations

The ordered notebook was reconstructed from the completed Kaggle workflow and syntax checked. It has not yet been rerun from a new empty Kaggle session. Before claiming independent one-click reproduction, run it once from a fresh session and confirm that the final tables match the included results.

