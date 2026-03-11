# ChemLM - a domain adaptable language model

### Domain adaptable language modeling of chemical compounds identifies potent pathoblockers for Pseudomonas aeruginosa

Computational techniques for predicting molecular properties are emerging as pivotal components for streamlining drug development, optimizing
time, and financial investments. Here, we introduce ChemLM, a transformer language model-based approach for this task. ChemLM further
leverages self-supervised domain adaptation on chemical molecules to enhance its predictive performance across new domains of interest.
Within the framework of ChemLM, chemical compounds are conceptualized as sentences composed of distinct chemical ‘words’, which are
employed for training a specialized chemical language model. In the standard benchmark datasets, ChemLM has either matched or surpassed
the performance of current state-of-the-art methods. Furthermore, we evaluated the effectiveness of ChemLM in identifying highly potent
pathoblockers targeting Pseudomonas aeruginosa (PA), a pathogen that has shown an increased prevalence of multidrug-resistant strains and has
been identified as a critical priority for the development of new medications. ChemLM demonstrated significantly higher accuracy in identifying
highly potent pathoblockers against PA when compared to state-of-the-art approaches. An intrinsic evaluation demonstrated the consistency of
the chemical language model’s representation concerning chemical properties. Our results from benchmarking, experimental data, and intrinsic
analysis of the ChemLM space confirm the wide applicability of ChemLM for enhancing molecular property prediction within the chemical domain.

![Image](/Figures/Chemlm_overview.png)	

The described approach is in [Chemrxiv](https://chemrxiv.org/engage/chemrxiv/article-details/657cb14de9ebbb4db9fa0e13).

## Replicate environment:
To replicate the conda environment of our approach
```
conda env create --file env.yml
conda activate chemlm_env
```

For lipschitz constant and script lipschitz_analysis.py use:
```
conda env create --file pval_env.yml
conda activate pval_env
```


To replicate the environment for the graph neural networks, type the following commands:
```
conda env create --file chemlm_env_requirements.yml
conda activate comparison_graphs
pip install tensorflow==2.4.0

conda install numpy==1.21.0
conda install -c conda-forge rdkit==2022.09.1
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch 
```

in case of ssl related errors ssl in an Ubuntu OS, run:
```
sudo update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs
```

## Data
Data that were used for the development and the evaluation of our model are provided.

Train, validation and test sets of the benchmark datasets are located in [data/benchmark](https://github.com/hzi-bifo/ChemLM/tree/main/data/benchmark).

The dataset that was used for intrinsic evaluation (Lipschitz constant calculation and UMAP) can be found in [data/intrinsic](https://github.com/hzi-bifo/ChemLM/tree/main/data/intrinsic).

Experimental data have been presented in peer-reviewed journals or patent applications and are detailed in Supplementary Table 4 and are not available here.

## Models 
The trained models including the pretrained and the tokenizer can be found in [Huggingface](https://huggingface.co/gkallergis).
Please download the models from that repo. 
Models that end with "_da" are the domain adapted versions and models with "hp_opt" for hyperparameter optimization. Model named as 'bbbp_intrinsic' is to be used for intrinsic evaluation in the relevant scripts. 


## Code

To run the desired script, use the corresponding `.sh` file:
```bash
./script.sh
```
Each `.sh` file includes an example of the required arguments. All generated datasets are saved in the specified folders. It is advised to run the scripts on a GPU with high memory resources.

---

### Hyperparameter Optimization (`ft/`)
`hp_tune.py` performs hyperparameter optimization. All models are located in `models/hp_models` for reproducibility. Provide the path for the model, tokenizer, and desired save path in the corresponding `.sh` script. This script produces:
- `study_{datasetName}.pkl` — Optuna study
- `{datasetName}_optimization_results.txt` — Best hyperparameters and their importance

---

### Domain Adaptation
`training_domain_adaptation.py` trains a domain-adapted model and produces:
- `model_{datasetName}_{augmentationNumber}` — Trained model
- `time_{datasetName}_{augmentationNumber}.txt` — Training time log

### Evaluation
The benchmark evaluation(evaluate.py) and pretraining scripts are located in the same folder. To perform evaluation, use the domain-adapted model from the `models/` folder. Benchmark evaluation produces:
- `results_{datasetName}_{modelType}.txt` — Evaluation results
- `preds_{datasetName}_{modelType}.txt` — Model predictions
- `labels_{datasetName}_{modelType}.txt` — True labels

---

### Visualization (`visualization/`)
| Script | Description |
|---|---|
| `Bench_plot.ipynb` | Dotplot figure for model comparison on benchmark datasets |
| `Molb_chemlm_analysis.py` | Violinplots (Fig. 5b) and statistical comparison (Table 3) |
| `viz_figures.py` | Figures 2 and 3b — only a save path is required |
| `ratio_analysis.py` | Figure 5a and Table 2 — only a save path is required |
| `intrinsic_vis.py` | UMAP plots (Fig. 5c), SI heatmap, and ChemLM/random space analysis files — see `.sh` for required models and files |

---

### Auxiliary (`aux_folder/`)
`bin_calc.py` reproduces evaluation metric scores from prediction and label files. This script was used to generate the results tables.

---

### Experimental (`experimental/`)
| Script | Description |
|---|---|
| `Clustering_sets.py` | Hierarchical folds, dataset generation, and Table 5 (SI). Saves folds as `hierarchical_fold_{foldNumber}.csv` |
| `experimental_evaluation.py` | Model evaluation on hierarchical folds. Saves results as `cv_experimental.txt` |

## How to Use for a New Dataset

For each step of the process, please use the arguments specified in the corresponding `.sh` script.

1. **Domain Adaptation** — Run `training_domain_adaptation.py` to perform domain adaptation. Train several models using high augmentation numbers. The desired augmentation number must be provided as an argument.

2. **Hyperparameter Optimization** — Run `hp_tune.py` to find the four main hyperparameters:
   - Number of augmentations
   - Layers
   - Attention heads
   - Embeddings type

   Given enough resources, exploring the ideal learning rate and batch size is also advised. Feel free to experiment with other parameters as well. Best hyperparameters will be stored in the designated `save_path` along with an Optuna study.

3. **Evaluation** — Run `evaluate.py` with the selected hyperparameters according to the `.sh` script.


## Comparison Models

---

### Graph Neural Networks
Implemented using the [DeepChem](https://deepchem.io/) library. Code is located in `code/comparison/` as `comparison.py`. To run the code, replicate and activate the corresponding environment.

---

### MolBERT
Downloaded and used from the [MolBERT repository](https://github.com/BenevolentAI/MolBERT). Modifications were made only to edit the data path and output files. For intrinsic evaluation, the modification script is located in [`code/aux_code/`](https://github.com/hzi-bifo/ChemLM/tree/main/code/aux_code). Cloning the corresponding repository is required.

---

### MolFormer
Downloaded and used from the [MolFormer repository](https://github.com/IBM/molformer). MolFormer is included in the `code/` folder as a `.zip`. 

---

### ChemBERTa
We used the `PubChem10M_SMILES_BPE_180k` model from Hugging Face. The evaluation script (`chemberta_bnch.ipynb`) is located in the `comparison/` folder.

---
## Results
We provide files that were used to generate the figures and the tables in the manuscript. 

Ratio distribution for ChemLM, MolBERT and random space are located in [results/ratio_distributions](https://github.com/hzi-bifo/ChemLM/tree/main/results/lipschitz_distributions). As mentioned above we downloaded and deployed MolBERT for benchmark comparison. We modified one of its scripts to get the necessary results for the intrinsic evaluation. In  [code/aux_code](https://github.com/hzi-bifo/ChemLM/tree/main/code/aux_code) we show how we modified the finetune.py file of the package molbert/models and how we used the script run_molbert_lip.py to get the results, having trained a molbert model on the bbbp data first. 


The labels and the predictions of the models that we report in the manuscript are located in [results/experimental_files](https://github.com/hzi-bifo/ChemLM/tree/main/results/experimental_files).

## Citations

Please cite:

@misc{Kallergis2023,    
author = {Kallergis, G and Asgari, E and Azarkhalili, B and Hirsch, A and Mchardy, A C},   
title = {{Domain adaptable language modeling of chemical compounds identifies potent pathoblockers for Pseudomonas aeruginosa}},    
year = {2023}    
}
