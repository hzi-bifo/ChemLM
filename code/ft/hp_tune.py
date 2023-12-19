import torch
import pandas as pd
import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn import preprocessing,metrics
import random
import torch
from argparse import ArgumentParser
from transformers import set_seed
import os
import torch.nn as nn
from transformers import RobertaModel,RobertaConfig
import gc
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import joblib
import logging
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

def load_data(dataset):
  data=pd.read_csv(dataset,sep='\t')
  #print(data)
  smiles=data.loc[:,'smiles'].values.tolist()
  labels=data.loc[:,'y'].values.tolist()
  labels_int=[float(label) for label in labels]
  #print('Loaded data')
  return smiles,labels_int

def create_dataloaders(inputs, masks, labels, batch_size,dev):
    #print(inputs)
    g = torch.Generator(device='cpu')
    g.manual_seed(0)

    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
    def seed_worker(worker_id):
        worker_seed = 10
        np.random.seed(worker_seed)
        np.seed(worker_seed)    
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, generator=g, worker_init_fn=seed_worker)
    return dataloader

def enc(smiles,tokenizer_pa):
    tokenizer=RobertaTokenizer.from_pretrained('{}'.format(tokenizer_pa))
    encoded = tokenizer(text=smiles,add_special_tokens=True,padding='longest',return_attention_mask=True)
    return encoded['input_ids'],encoded['attention_mask']

class Chemlm_model(nn.Module):
    
    def __init__(self, dt_n,drop_rate,an, att, n_layers, tt,seed_,emb_type, model_path):
        
        super(Chemlm_model, self).__init__()
        self.type=emb_type
        self.task=tt
        if tt=='clf':
            D_out=2
        else:
            D_out=1
        D_in = 768
        set_seed_all(seed_)
        config=RobertaConfig.from_pretrained('{}/bbbp_unsup_full_{}_all_skf_1023'.format(model_path,an),output_hidden_states=True,output_attentions=True,num_hidden_layers = n_layers,num_attention_heads = att)
        self.roberta = RobertaModel.from_pretrained('{}/bbbp_unsup_full_{}_all_skf_1023'.format(model_path,an),config=config)        
        for param in self.roberta.parameters():
           param.requires_grad=False 
        self.dense = nn.Linear(D_in, D_in)
        self.dropout = nn.Dropout(drop_rate)
        self.out_proj = nn.Linear(D_in, D_out)

    def forward(self, input_ids, attention_masks):
        outputs = self.roberta(input_ids, attention_masks)
        #print(outputs)
        if self.type=='pooling':
            class_label_output = outputs[1]
        elif self.type=="last":
            class_label_output = outputs[0][:,0]
        elif self.type=="last_mean":
            last_hidden_state = outputs[0]
            class_label_output = torch.mean(outputs[0],axis=1)
        elif self.type=="last_sum":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.sum(all_hidden_states,0)
            class_label_output=torch.mean(mean_w,1)         
        elif self.type=="mean_0":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.mean(all_hidden_states,0)
            class_label_output=mean_w[:,0]
        elif self.type=="mean_mean":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.mean(all_hidden_states,0)
            class_label_output=torch.mean(mean_w,1)
        elif self.type=="mean_sum":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.sum(all_hidden_states,0)
            class_label_output=torch.sum(mean_w,1)
        elif self.type=="sum_0":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.sum(all_hidden_states,0)
            class_label_output=mean_w[:,0]
        elif self.type=="sum_mean":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.sum(all_hidden_states,0)
            class_label_output=torch.mean(mean_w,1)
        elif self.type=="sum_sum":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.sum(all_hidden_states,0)
            class_label_output=torch.sum(mean_w,1)
            #print(class_label_output.shape)        
        #class_label_output = outputs[1]
        x = self.dropout(class_label_output)
        x = self.dense(class_label_output)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)        
        return x   

def define_model(trial, model_p, dt):
	n_layers=trial.suggest_int('n_layers', 4,12,step=4)
	n_att_heads=trial.suggest_int('n_att', 8,16, step=4)
	t_embeddings=trial.suggest_categorical('embds', ['pooling','last','last_mean','last_sum','mean_0','mean_mean','mean_sum','sum_0','sum_mean','sum_sum'])
	augm_number=trial.suggest_categorical('augmentation',[0,5,10,15,20,25,40,60,80,100])
	return 	Chemlm_model( model_path = model_p, dt_n=dt , emb_type = t_embeddings, drop_rate=0.1,an=augm_number,tt='clf',n_layers=n_layers,att=n_att_heads,seed_=3407)

def objective(trial,dataset_n, model_path, tokenizer_path, train_dt, valid_dt):
    dataset_n='bbbp'
    gc.enable()
    joblib.dump(study, 'study_{}.pkl'.format(dataset_n),compress=1)
    seed=3407
    set_seed_all(seed)  
    if torch.cuda.is_available():       
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    batch_size= 16
    epochs=7
    lrate= 0.0005
    path='{}'.format(train_dt)
    path_vl='{}'.format(valid_dt)
    smiles_tr,labels_tr=load_data(path)
    smiles_val,labels_val=load_data(path_vl)
    inputs_tr, masks_tr=enc(smiles_tr,tokenizer_path)
    inputs_val, masks_val=enc(smiles_val,tokenizer_path)
    train_dataloader=create_dataloaders(inputs_tr, masks_tr, labels_tr, batch_size,device)
    val_dataloader=create_dataloaders(inputs_val, masks_val, labels_val, batch_size,device)
    model = define_model(trial, model_path, dataset_n).to(device)
    optimizer = AdamW(model.parameters(),
	              lr=lrate,                 eps=1e-8,weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
	             num_warmup_steps=0, num_training_steps=total_steps)
    loss_function = nn.CrossEntropyLoss()
    #print(augm_number)
    for epoch in range(epochs):
    	total_loss=0
    	model.train()
    	for step,batch in enumerate(train_dataloader):
    		batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
    		model.zero_grad()
    		outputs=model(batch_inputs, batch_masks)
    		loss=loss_function(outputs,batch_labels.long())
    		total_loss+=loss.item()
    		loss.backward()
    		clip_grad_norm_(model.parameters(), 2)
    		optimizer.step()
    		scheduler.step()
    	model.eval()
    	val_loss=0
    	predictions, batch_l=[],[]
    	for step_vl,batch_vl in enumerate(val_dataloader):
    		batch_inputs_vl, batch_masks_vl, batch_labels_vl = tuple(b.to(device) for b in batch_vl)
    		with torch.no_grad():
    			outputs_vl=model(batch_inputs_vl, batch_masks_vl)
    			preds = torch.argmax(outputs_vl, dim=-1)
    			predictions.append(preds.tolist())
    			batch_l.append(batch_labels_vl.tolist())
    	outputs_flatten=[item for sublist in predictions for item in sublist]
    	labels_flatten= [item for sublist in batch_l for item in sublist]
    	f1=metrics.f1_score(labels_flatten,outputs_flatten,average='macro')
    	trial.report(f1, epoch)
    	if trial.should_prune():
    		raise optuna.exceptions.TrialPruned()
    	gc.collect()
    del model
    del outputs_flatten
    del labels_flatten
    gc.collect()
    return f1


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)      
    parser.add_argument('--valid_path', type=str, required=True)       
    parser.add_argument('--model_path', type=str, required=True)          
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args= parser.parse_args()
    model_path=args.model_path
    tokenizer_path = args.tokenizer_path
    path_t = args.train_path
    path_v= args.valid_path
    save_path = args.save_path
    dataset_n = args.dataset
    logging.basicConfig(level='ERROR')
    study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=10))
    study.optimize(lambda trial: objective(trial, dataset_n, model_path, tokenizer_path, path_t, path_v ), n_trials=1,  callbacks=[lambda study, trial: gc.collect()])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    results = optuna.importance.get_param_importances(study)
    print(results)
    f = open('{}/{}_optimization_results.txt'.format(save_path, dataset_n),'w')
    for key, value in trial.params.items():
        f.write("{}: {}".format(key, value))
    f.write('\n\n{}'.format(results))
    f.close()  
