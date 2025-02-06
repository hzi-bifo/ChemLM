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
To run the desired script, use the corresponding ".sh" file as such:
```
./script.sh
```

In ft foder are located the hyperparameter optimization of the models. We provide multiple domain-adapted models for bbbp dataset in Huggingface repo. Those are to be used for reproducibility of hyperparameter optimization script. Please provide the path for the model, the tokenizer and the desired path to save the results in every '.sh' script.
 The evaluation on the benchmark datasets and the pretraining file are also located in the same folder. To perform evaluation, use the domain-adapted model from Huggingface. 

In visualization folder are located the scripts for the tables and the figures of the manuscript. 
Using bin_calc.py in aux folder can be reproduced the evaluation metrics scores from the files that contain the predictions and the labels of each model. This script was used to create the tables. 

It is advised to run the scripts on GPU with high memory resources.


## Code of graph neural networks and language models

Graph neural networks are implemented using deepchem library. The code is located in [code/comparison](https://github.com/hzi-bifo/ChemLM/tree/main/code/comparison). To run the code, please replicate and activate the corresponding environment. The outcome of these models and the labels can be found in the following results section. 

MolBERT was downloaded and used from the corresponding  [Github repo](https://github.com/BenevolentAI/MolBERT).

Molformer was downloaded and used from the corresponding  [Github repo](https://github.com/IBM/molformer).

We used the ChemBERTa model "PubChem10M SMILES BPE 180k" model from Hugging Face, and the evaluation script (chemberta_bnch.ipynb) is located at the comparison folder.

## Results
We provide files that were used to generate the figures and the tables in the manuscript. 

Lipschitz distribution for ChemLM, MolBERT and random space are located in [results/lipschitz_distributions](https://github.com/hzi-bifo/ChemLM/tree/main/results/lipschitz_distributions). As mentioned above we downloaded and deployed MolBERT for benchmark comparison. We modified one of its scripts to get the necessary results for the intrinsic evaluation. In  [code/aux_code](https://github.com/hzi-bifo/ChemLM/tree/main/code/aux_code) we show how we modified the finetune.py file of the package molbert/models and how we used the script run_molbert_lip.py to get the results, having trained a molbert model on the bbbp data first. 

The results of hyperparameters optimization are reported in [results/optimization_files](https://github.com/hzi-bifo/ChemLM/tree/main/results/optimization_files).

The labels and the predictions of the models that we report in the manuscript are located in [results/experimental_files](https://github.com/hzi-bifo/ChemLM/tree/main/results/experimental_files).

## Citations

Please cite:

@misc{Kallergis2023,    
author = {Kallergis, G and Asgari, E and Azarkhalili, B and Hirsch, A and Mchardy, A C},   
title = {{Domain adaptable language modeling of chemical compounds identifies potent pathoblockers for Pseudomonas aeruginosa}},    
year = {2023}    
}
