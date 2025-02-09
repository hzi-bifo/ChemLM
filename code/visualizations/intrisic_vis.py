import pandas as pd
import numpy as np
import torch
import tqdm
import argparse
from sklearn import preprocessing
import umap
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import random
from transformers import set_seed
from transformers import RobertaTokenizer, RobertaModel,RobertaConfig
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
import os
import gc
import itertools
#This script produces the UMAP plots for figure and the  Supplementary Fiugre 1 that deomnstrates the mean attention a token gets from the others in different layers. 

#creates the umap plots for figure 4
def umap_plot(embeddings,labels,dataset, title,nn,s_path):
	plt.cla()
	plt.clf()
	n_components=2
	num_embeddings=[t.detach().cpu().numpy() for t in embeddings]
	y = umap.UMAP(n_components=2, n_neighbors=nn, metric='correlation',random_state=4, verbose=True).fit_transform(num_embeddings)
	if dataset=='psa' or dataset =='mw':
		labels = np.log(labels)
	else:
		labels = labels
	min_l = np.min(labels)
	max_l = np.max(labels)
	r = np.array(labels)
	rang = np.unique(r)
	si=[100 for s in range(len(labels))]
	continuous_cmap = plt.get_cmap('plasma')
	num_categories = max(labels)
	plt.scatter(y[:, 0], y[:, 1], c=labels, cmap=continuous_cmap, alpha=0.7, s=si)
	plt.xlabel('UMAP 1',fontsize=24)
	plt.ylabel('UMAP 2',fontsize=24)
	cbar = plt.colorbar(orientation='horizontal')
	cbar.set_label('{}'.format(title),labelpad=50,fontsize=30, rotation=0)
	cbar.ax.tick_params(labelsize=18)
	plt.tight_layout()
	figure = plt.gcf()
	figure.set_size_inches(25, 20)
	plt.xticks([])
	plt.yticks([])
	#print('Made it here')
	plt.savefig('{}/umap_{}_{}'.format(s_path, nn, dataset))
	print('Created UMAP plot for: {}'.format(title))

# creates figure Supplementary Figure 1	
def plt_heatmap(df,layers,s_path):
	print(df)
	tokens=df.tokens.to_list()
	df=df.set_index('tokens')
	ax = sns.heatmap(df, annot=False, linewidths=0.5, cmap = plt.cm.plasma,cbar_kws={'shrink':0.5}) #cmap=sns.light_palette("#79C",as_cmap=True),cbar_kws={'shrink':0.5})
	cbar = ax.collections[0].colorbar
	cbar.ax.tick_params(labelsize=20)
	cbar.set_label(label= 'Mean Attention Weight', 	labelpad=50)
	ax.figure.axes[-1].yaxis.label.set_size(30)
	plt.xlabel('Attention Layers',fontsize=35)
	ax.set_xticklabels(layers, rotation=45)
	ax.set_yticklabels(tokens, rotation=0)
	ax.xaxis.set_tick_params(labeltop=True,labelsize=24)
	ax.yaxis.set_tick_params(labelbottom=True,labelsize=24)
	ax.yaxis.set_tick_params(labeltop=False)
	ax.xaxis.set_tick_params(labelbottom=False)

	plt.ylabel('Tokens of SMILES',fontsize=35)
	o_patch = mpatches.Patch( color='None',label='O: atom of oxygen ')
	c_patch = mpatches.Patch( color='None',label='C: atom of carbon ')
	n_patch = mpatches.Patch( color='None',label='N: atom of nitrogen ')
	car_patch = mpatches.Patch( color='None',label='c: atom of carbon in an aromatic ring')
	par_patch = mpatches.Patch( color='None',label='(): parentheses indicate the start and the end of a branch')
	num_patch = mpatches.Patch( color='None',label='Numbers: indicate the ring')
	cls_patch = mpatches.Patch( color='None',label='<s>: indicates the start of sequence token')
	end_patch = mpatches.Patch( color='None',label='</s>: indicates the end of sequence token')

	plt.subplots_adjust(top=0.99)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
	                 box.width, box.height * 0.8])

	ax.legend(title='Meaning of symbols in tokens',fontsize=22,  title_fontsize=30,handles=[o_patch,c_patch,n_patch,car_patch,par_patch,num_patch,cls_patch,end_patch], bbox_to_anchor=(1.3, -0.1),
	          fancybox=True, shadow=True, ncol=3)
	figure = plt.gcf()
	plt.tight_layout()
	figure.set_size_inches(25, 20)
	plt.savefig('{}/heatmap_smiles'.format(s_path))	

def create_embeddings_heatmap_att(model,tokenizer,compounds_func,flag_layer,prop,device):
	embeddings=[]
	labels=[]
	counter_failed=0
	for ind,compound in compounds_func.iterrows():
		if type(compound.smiles)==float:
			counter_failed+=1
			continue
		inputs=tokenizer(compound.smiles, return_tensors="pt")
		ex=tokenizer.tokenize(compound.smiles)
		ex.insert(0,'<s>')
		ex.append('</s>')
		if len(inputs['input_ids'][0])>14 or len(inputs['input_ids'][0])<8:
			continue
		else:
			print('FOUND')
			out=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(compound.smiles))
		outputs = model(inputs['input_ids'].to(device) , inputs['attention_mask'].to(device))
		h, l=[], []
		hidden_state=(outputs[3][:])
		for i in range(len(hidden_state)):
			mk=hidden_state[i]
			k=mk[0]
			m=torch.mean(k, dim=0)
			m=torch.mean(m,dim=0)
			h.append(m)
		num_embeddings=[t.detach().cpu().numpy() for t in h]
		return ex,len(hidden_state), num_embeddings

def create_embeddings_heatmap(model,tokenizer,compounds_func,flag_layer,prop):
	embeddings=[]
	labels=[]
	counter_failed=0
	for ind,compound in (compounds_func.iterrows()):
		if type(compound.Smiles)==float:
			counter_failed+=1
			continue
		inputs=tokenizer(compound.Smiles, return_tensors="pt")
		ex=tokenizer.tokenize(compound.Smiles)
		ex.insert(0,'CLS')
		ex.append('END')
		if len(inputs['input_ids'][0])>14 or len(inputs['input_ids'][0])<8:
			continue
		else:
			out=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(compound.Smiles))
		try:
			outputs = model(**inputs)
		except:
			counter_failed+=1
			continue 
		h=[]
		hidden_state=(outputs[2])
		all_hidden_states = torch.stack(hidden_state)
		mean_w = torch.sum(all_hidden_states,0)
		class_label_output=mean_w[:,0]
		return ex,len(hidden_state), class_label_output.detach().numpy() 

def lipschitz(vec,labels):
	z1=zip(vec,labels)
	k=[]
	for pair in itertools.combinations(z1,2):
		b=pair[0][0].squeeze()
		a=pair[1][0].squeeze()
		#print(a)
		vec_diff=sum(((a - b)**2).reshape(768)).sqrt()
		y_diff=abs(pair[0][1]-pair[1][1])
		try:
			norm=(y_diff/float(vec_diff))
			k.append(norm.item())
		except:	
			pass
	return np.mean(k)

#shuffles the labels for the random space
def random_space(labels, arg_seed):
	b = labels[:]
	random.shuffle(b)
	return b

def att_heat_gen(model_f, tokenizer,df_tr, device):
	tokens,attention, num_embeddings=create_embeddings_heatmap_att(model_f,tokenizer,df_tr,'all_mean','ha',device)
	num_layers=[str(i) for i in range(attention)]
	df=pd.DataFrame()
	df['tokens']=tokens
	df['layer 0']=num_embeddings[:][0]
	df['layer 1']=num_embeddings[:][1]
	df['layer 2']=num_embeddings[:][2]
	df['layer 3']=num_embeddings[:][3]
	df['layer 4']=num_embeddings[:][4]
	df['layer 5']=num_embeddings[:][5]
	df['layer 6']=num_embeddings[:][6]
	df['layer 7']=num_embeddings[:][7]
	layers=['Layer {}'.format(k) for k in range(attention)]
	return df, layers

#set all seeds for multiple libraries that might affect the results
def set_seed_all(seed): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    random.seed(seed)
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class Chemlm_model(nn.Module):
    def __init__(self, dt_n,drop_rate,an, att, n_layers, tt,seed_,emb_type, mo_path,flag):   
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
        if self.flag_ == 1:
        	return outputs
        class_label_output = outputs[0][:,0]
        return class_label_output   

def enc(smiles, tokenizer_path):
    tokenizer=RobertaTokenizer.from_pretrained(tokenizer_path)
    encoded = tokenizer(text=smiles,add_special_tokens=True,padding='longest',return_attention_mask=True)
    return encoded['input_ids'],encoded['attention_mask']

def create_dataloaders(inputs, masks, labels, batch_size,dev):
    g = torch.Generator(device='cpu')
    g.manual_seed(0)
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor   )
    def seed_worker(worker_id):
        worker_seed = 10
        np.random.seed(worker_seed)
        np.seed(worker_seed)    
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False)#, generator=g, worker_init_fn=seed_worker)
    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)             
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--function', type=int, required=True)
    parser.add_argument('--mpath', type=str, required=True)
    parser.add_argument('--rounds_num', type=int, required=False,default=-1)
    parser.add_argument('--samples_num', type=int, required=False, default=-1)

    args= parser.parse_args()
    save_p = args.save_path
    dataset_n=args.dataset
    m_path = args.model_path
    t_path = args.tokenizer_path
    mm_path = args.mpath
    flag=args.function
    n_rounds = args.rounds_num
    n_samples=args.samples_num
    if flag>2:
    	print('Acceptable: function values 0, 1, 2 (--function)')
    	return
    gc.enable()
    seed=3407
    set_seed_all(seed)        
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    current_directory = os.getcwd()
    target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
    target_f = os.path.abspath(os.path.join(target_folder, 'data/intrinsic/'))
    file_path = os.path.join(target_f, dataset_n)
    final = pd.read_csv(file_path, sep=',', header=0,  quotechar='"', error_bad_lines=False)
    sm_tokens,masks=enc(final.smiles.tolist(), t_path)
    model = Chemlm_model(dt_n='bbbp',drop_rate=0.1,an=80, att=12,n_layers= 8, tt='clf',seed_=3407,emb_type='last', mo_path= m_path, flag=flag)
    if device.type=='cpu':
    	model.load_state_dict(torch.load('{}'.format(mm_path),map_location=torch.device('cpu')))
    else:
    	model.load_state_dict(torch.load('{}'.format(mm_path)))

    model.to(device)
    model.eval()

    properties=['psa', 'mw','ha','hd', 'qed','aromatic']
    titles=['Polar surface area (Å2)','Molecular weight (Da)', 'Number of \nhydrogen-bond acceptors', 'Number of \nhydrogen-bond donors', 'Quantitative Estimation \nof Drug Likeness (QED)','Number of\naromatic bonds']
    if flag==0:
    	neighbors_list=[90]
    	for neighbors in neighbors_list:
    		print('Creating UMAP plots...\n\n')
	    	for prop, title in zip(properties, titles):
	    		lab, embds=[],[]
	    		lab_init_l=final[prop].tolist() #lab_init.tolist()
	    		dataloader=create_dataloaders(sm_tokens,masks,lab_init_l,1,device)
	    		for step,batch in enumerate(dataloader):
	    			batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
	    			with torch.no_grad():
	    				outputs=model(batch_inputs, batch_masks)
	    				lab.extend(batch_labels)
	    				embds.extend(outputs)
	    		umap_plot(embds,lab,prop,title, neighbors, save_p)
	    		del lab
	    		del embds
	    		del dataloader

    elif flag==1:
	    tokenizer=RobertaTokenizer.from_pretrained(t_path)
	    df_a, layers_a = att_heat_gen(model,tokenizer,final,device)
	    plt_heatmap(df_a,layers_a)
    elif flag==2:
    	if n_samples == -1 or n_rounds==-1:
    		print('Provide number of samples(--samples_num) and number of rounds (--rounds_num)')
    		return
    	range_seeds = 100
    	seeds = [k for k in range(range_seeds)]
    	for prop in properties:
    		#f=open('lipschitz_random_{}_200n_100vals_revisited_upd.txt'.format(prop),'w')
    		#f=open('/vol/projects/gkallerg/lpmcr/lipschitz/lipschitz_random_{}_200n_100rounds.txt'.format(prop),'w')
    		f=open('{}/chemlm_random_{}_{}n_{}rounds_f.txt'.format(save_p, prop, n_samples,n_rounds),'w')

    		for seed in seeds:
    			k_list=[]
	    		k_random_list=[]
	    		counter = 0
	    		lab, embds=[],[]
	    		final_df= final.sample(n=200, axis = 0, random_state = seed)
    			# final_df.to_csv('/vol/projects/gkallerg/lpmcr/lipschitz/lipschitz_{}_{}_random.csv'.format(prop,seed)) 	    		
    			final_df.to_csv('{}/lipschitz_{}_{}_random_f.csv'.format(save_p, prop,seed)) 	    		    			
	    		lab_init_l=final_df[prop].tolist() #lab_init.tolist()	    		
	    		sm_tokens,masks=enc(final_df.smiles.tolist())
	    		dataloader=create_dataloaders(sm_tokens,masks,lab_init_l,1,device)
	    		for step,batch in enumerate(dataloader):
	    			batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
	    			with torch.no_grad():
	    				outputs=model(batch_inputs, batch_masks)
	    				lab.extend(batch_labels)
	    				embds.extend(outputs)#f.write('\n\nProperty: {}'.format(prop))
	    		r_ls = random_space(lab, seed)
    			k_r=lipschitz(embds,r_ls)
    			f.write('{}|'.format(k_r))
	    		k=lipschitz(embds,lab)
	    		f.write('{}\n'.format(k))
	    		print('{}|{}'.format(k, k_r))
	    		del final_df
	    		del sm_tokens
	    		del lab_init_l
    		f.close()
	    	print('\n\n\n')
	    	print('Closed file')    

if __name__=='__main__':
	main()
