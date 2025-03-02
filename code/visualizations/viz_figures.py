import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn import metrics
from scipy import stats
import os
#script that implements Figure 2, 3b, the p-values and the distribution plots for Lipschitz constant.

#Figure 1, using the values from the generated file of hp_opt.py
#values reported in the corresponding file in results/optimization
def boxplot_hyper(save_path):
	p_=sns.set_palette(['#b3b3b3', '#008176','#0000a7','#eecc16'])
	#p_ = sns.color_palette("deep", 4)
	feat=['No. of Layers','Embeddings', 'No. of Attention\nheads', 'Augmentation\nNumber','No. of Layers','Embeddings', 'No. of Attention\nheads', 'Augmentation\nNumber','No. of Layers','Embeddings', 'No. of Attention\nheads', 'Augmentation\nNumber']
	imp =  [0.004,0.831, 0.019, 0.144,  0.09, 0.75,0.055, 0.1, 0.038, 0.439,  0.051, 0.47]
	g= sns.boxplot(y = imp , x= feat, palette = p_)
	plt.xlabel('Hyperparameter',size=40,labelpad=15)
	plt.ylabel('Importance',size=40,labelpad=15)
	plt.xticks(fontsize=35)
	plt.yticks(fontsize=35)
	fig = plt.gcf()
	fig.set_size_inches(20, 15)
	plt.tight_layout(w_pad=0.05)
	plt.savefig('{}/boxplot_hyperparameters'.format(save_path), dpi=400)
	plt.close()


def boxplot_experimental(save_path):
	p2_ = sns.set_palette(['#fef090','sandybrown','#bf7467','#abd9e9', '#74add1','#4575b4','#d73027'])#a50026'])
	model= ['MolBERT','MolBERT','MolBERT','MolBERT','MolBERT', 'MolFormer','MolFormer', 'MolFormer', 'MolFormer', 'MolFormer', 'ChemBERTa','ChemBERTa','ChemBERTa','ChemBERTa','ChemBERTa', 'GAT', 'GAT','GAT','GAT','GAT','MPNN','MPNN','MPNN','MPNN','MPNN','GCN','GCN','GCN','GCN','GCN','ChemLM','ChemLM','ChemLM','ChemLM','ChemLM']
	f1 = [ 0.435, .495, .713, 0.441, .6,     0.47, 0.4, 0.29, 0.24, 0.33,       0.47, 0.4, 0.29, 0.17, 0.33,       0.529,  0.563,0.565, 0.441,0.697,  0.469,0.541,0.734,0.604,0.792,                     0.571,0.519,0.612,0.842,0.333,       0.458,0.760,0.914,0.927,0.898]
	folds=['5' for _ in range(len(f1))]
	df = pd.DataFrame(list(zip(model,f1, folds)),columns=['model', 'f1','folds'])
	g=sns.boxplot(x=df['model']  ,y=df['f1'], palette=p2_)
	#median_values = df.groupby("Models")["F1-score"].mean().reset_index()
	p3 = sns.set_palette(['#fee090','#fed290','#fec090','#abd9e9', '#74add1','#4575b4','#d73027'])#a50026'])
	sns.swarmplot(data = df, x='model', y='f1',  size=15, color='darkgray' )
	plt.ylabel('F1-score', size=35)
	g.tick_params(axis='x', labelsize=27)
	plt.xlabel('Models', size=35)
	g.set_xticklabels(['MolBERT','MolFormer','ChemBERTa', 'GAT','MPNN',  'GCNN','ChemLM'])
	g.tick_params(axis='y', labelsize=27)
	figure = plt.gcf()
	figure.set_size_inches(20, 15)
	#plt.show()
	plt.savefig('{}/boxplot_experimental.png'.format(save_path), dpi=400)
	plt.close()

if __name__ =='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_path', type=str, required=True)
	args= parser.parse_args()
	save_p = args.save_path
	boxplot_hyper(save_p)
	boxplot_experimental(save_p)

