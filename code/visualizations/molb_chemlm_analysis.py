import pandas as pd
from transformers import RobertaTokenizer, RobertaModel,RobertaConfig
import tqdm
import numpy as np
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing,metrics
import random
from transformers import set_seed
import os
import torch.nn as nn
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import scipy
import argparse

def set_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
#    torch.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    random.seed(seed)
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class Chemlm_model(nn.Module):
    def __init__(self, dt_n,drop_rate,an, att, n_layers, tt,seed_,emb_type, mo_path):   
        super(Chemlm_model, self).__init__()
        self.type=emb_type
        self.task=tt
        if tt=='clf':
            D_out=2
        else:
            D_out=1
        D_in = 768
        inter_med=384
        set_seed_all(seed_)
        config=RobertaConfig.from_pretrained('{}'.format(mo_path),output_hidden_states=True,output_attentions=True,num_hidden_layers = n_layers,num_attention_heads = att)
        self.roberta = RobertaModel.from_pretrained('{}'.format(mo_path),config=config)  
        for param in self.roberta.parameters():
           param.requires_grad=False
        self.dense = nn.Linear(D_in, D_in)
        self.dropout = nn.Dropout(drop_rate)
        self.out_proj = nn.Linear(D_in, D_out)
        self.flag_ = flag

    def forward(self, input_ids, attention_masks):
        outputs = self.roberta(input_ids, attention_masks)
        class_label_output = outputs[0][:,0]
        return class_label_output   

def load_data_n(dataset):
	data = pd.read_csv(dataset, sep=',', header=0,  quotechar='"', error_bad_lines=False)
	print(data)
	ec = data.loc[:,['smiles', 'psa', 'qed', 'mw']]
	return ec

def calc_distance(emb1, emb2 ): #calculates embeddings distance
	return np.linalg.norm(emb1.cpu() - emb2.cpu())

def calc_props(pr1, pr2): #calculates property distance
	return np.abs(pr1 - pr2)

def enc(smiles, t_path):
    tokenizer=RobertaTokenizer.from_pretrained(t_path)
    encoded = tokenizer(text=smiles,add_special_tokens=True,padding='longest',return_attention_mask=True, return_tensors='pt')
    return encoded['input_ids'],encoded['attention_mask']

def chemlm_emb(model, t_path, sm1, sm2, device):
	sm_tokens1,masks1 = enc(sm1, t_path)
	sm_tokens2,masks2 = enc(sm2, t_path)
	output1 = model(sm_tokens1.to(device),masks1.to(device))
	output2 = model(sm_tokens2.to(device), masks2.to(device))
	dist = calc_distance(output1, output2)
	return dist

def get_title_name(prop_name): #aux function for the titles in the corresponding plots
	if prop_name =='qed':
		prop = 'QED'
	elif prop_name =='psa':
		prop= 'PSA'
	elif prop_name =='mw':
		prop = 'Molecular Weight'
	return prop	

#check save path
def violin_plot(X , prop, save_path, prop2): #Violin plots for the comparison of ChemLM to MolBERT, Figure 5b, page 15
	prop_ = get_title_name(prop)
	figure(figsize=(12, 11), dpi=80)	
	vals = X+prop2
	models_ = ['ChemLM' for x in range(len(X))] + ['MolBERT' for y in range(len(prop2))] 
	dff = pd.DataFrame({'vals': vals, 'models':models_})
	sns.violinplot(data=dff, x="models", y="vals", palette = [ '#d73027', '#fee090'])
	plt.xlabel('Model', fontsize=40)
	plt.xticks(fontsize=36)
	plt.yticks(fontsize=32)
	plt.ylabel('Property distance', fontsize=36)	
	plt.title(f'{prop_}', fontsize=40)
	#plt.legend(loc='center left', bbox_to_anchor=(0.35, -0.1), prop={'size': 12})
	plt.tight_layout()
	plt.savefig(f'{save_path}/violinplot_{prop}.pdf')

	print('saved figure')

def read_file(prop,  model_name):
	current_directory = os.getcwd()
	target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
	target_f = os.path.abspath(os.path.join(target_folder, 'results/lipschitz_distributions/'))
	if model_name == 'chemlm':
		df = pd.read_csv(f'{target_f}/chemlm_chmb_{prop}.csv')
	else:
		df = pd.read_csv(f'{target_f}/molbert_chmb_{prop}.csv')
	return df

def tasks(prop,  save_path): 
	df = read_file(prop,  'chemlm')
	df_mol = read_file(prop,  'molbert')
	violin_plot( df.lip.tolist(), prop, save_path,  df_mol.lip.tolist())	
	stats(prop, df2, 'chemlm')
	stats(prop, df2_mol, 'molbert')

def stats(prop, df, model_name_): #calculates the statistical values for table 3, page 14. It is the comparison for ChemLM and MolBERT on Lipscitz compounds
	print(model_name_)
	vals = df.lip.tolist()
	print(f'Max value: {max(vals)}')
	print(f'Median value: {np.median(vals)}')
	print(f'std value: {np.std(vals)}')
	print(f'mad value: {scipy.stats.median_abs_deviation(vals)}')

def calc_lip(props,save_path, m_path, mm_path, t_path, chemlm_emb): #calculates the ratio of property to embeddings distance for molecular pairs. This is for the ChemLM approach. The same method has been used in MolBERT on their code. 
    if torch.cuda.is_available():
    	device = torch.device("cuda")
    	print("Using GPU.")
    else:
    	print("No GPU available, using the CPU instead.")
    	device = torch.device("cpu")
    current_directory = os.getcwd()
	target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
	target_f = os.path.abspath(os.path.join(target_folder, 'data/intrinsic'))	
    data = load_data_n('{target_f}/bbbp_comps_properties.csv')
    model = Chemlm_model(dt_n='bbbp',drop_rate=0.1,an=80, att=12,n_layers= 8, tt='clf',seed_=3407,emb_type='last', m_p = m_path)
    model.load_state_dict(torch.load(model_path_))
   if device.type=='cpu':
    	model.load_state_dict(torch.load('{}'.format(mm_path),map_location=torch.device('cpu')))
    else:
    	model.load_state_dict(torch.load('{}'.format(mm_path)))
    model.eval()

    df=pd.DataFrame(columns=['distance', 'diffs', 'lip'])
    for prop in props:
    	print(f'Property: {prop}')
    	con_max = -1
    	for i in tqdm.tqdm(range(len(data))):
    		for j in range(i):
	    		dist = chemlm_emb(model,t_path, data.loc[i].smiles, data.loc[j].smiles, device)
	    		abs_prop = calc_props(data.loc[i][prop], data.loc[j][prop])
	    		con = abs_prop/dist
	    		if con_max < con:
	    			con_max = con
	    			con_pair = (dist, abs_prop)
	    		df = df.append({'distance':dist, 'diffs':abs_prop, 'lip':con}, ignore_index=True)
    	df.to_csv(f'{save_path}/chemlm_chmb_{prop}.csv')
    	del df
    	gc.collect()

    	
if __name__=='__main__':
	props = [  'qed','psa', 'mw']
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)             
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--mpath', type=str, required=True)

    args= parser.parse_args()
    save_p = args.save_path
    m_path = args.model_path
    t_path = args.tokenizer_path
    mm_path = args.mpath
	for prop in props:
		print('----------------')		
		print(prop)
		tasks(prop, save_p)
