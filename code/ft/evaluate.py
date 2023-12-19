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
from argparse import ArgumentParser
from transformers import set_seed
import os
import torch.nn as nn
from transformers import RobertaModel,RobertaConfig
import gc


class ChemLM_model(nn.Module):
    
    def __init__(self, dt_n,drop_rate,an, att, n_layers, tt,seed_,emb_type, fr, model_path, t_path):
        
        super(ChemLM_model, self).__init__()
        self.type=emb_type
        self.task=tt
        if tt=='clf':
            D_out=2
        else:
            D_out=1
        D_in = 768
        
        set_seed_all(seed_)
        config=RobertaConfig.from_pretrained('{}'.format(model_path),output_hidden_states=True,output_attentions=True,num_hidden_layers = n_layers,num_attention_heads = att)
        self.roberta = RobertaModel.from_pretrained('{}'.format(model_path),config=config)       
        for param in self.roberta.parameters():
           param.requires_grad=fr
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
        elif self.type=="sum_mean":
            all_hidden_states = torch.stack(outputs[2])
            mean_w = torch.sum(all_hidden_states,0)
            class_label_output=torch.mean(mean_w,1)
        x = self.dropout(class_label_output)
        x = self.dense(class_label_output)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)        
        return x   

def set_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    set_seed(seed)

def early_stopping_f1(f1, previous_val_los,patience,epo,best_ep):
    break_flag=False
    triggers=-1
    if epo<3:
        return 0,break_flag,previous_val_los, -1  
    if f1<previous_val_los:
        if epo-best_ep>=patience:
            break_flag=True
            print('Early stopping activated')
    else:
        previous_val_los=f1
        best_ep = epo
    return triggers,break_flag,previous_val_los, best_ep

def load_data(dataset):
  data=pd.read_csv(dataset,sep='\t')
  smiles=data.loc[:,'smiles'].values.tolist()
  labels=data.loc[:,'y'].values.tolist()
  labels_int=[float(label) for label in labels]
  print('Loaded data')
  s = data.loc[:,'y'].value_counts()
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
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        np.seed(worker_seed)    
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, generator=g, worker_init_fn=seed_worker)
    return dataloader

def enc(smiles, t_p):
    tokenizer=RobertaTokenizer.from_pretrained('{}'.format(t_p))
    encoded = tokenizer(text=smiles,add_special_tokens=True,padding='longest',return_attention_mask=True)
    return encoded['input_ids'],encoded['attention_mask']

def validation(model_c,device,dataloader_vl,loss_function):
    model_c.eval()
    val_loss=0
    batch_l, predictions_prob,predictions =[],[],[]
    for step_vl,batch_vl in enumerate(dataloader_vl):
        batch_inputs_vl, batch_masks_vl, batch_labels_vl = tuple(b.to(device) for b in batch_vl)
        with torch.no_grad():
            outputs_vl=model_c(batch_inputs_vl, batch_masks_vl.type(dtype=torch.long))
            preds = torch.argmax(outputs_vl, dim=-1)
            predictions.append(preds.tolist())
            batch_l.append(batch_labels_vl.tolist())
            loss_vl=loss_function(outputs_vl,batch_labels_vl.type(torch.long))
            val_loss+=loss_vl.item()
    outputs_flatten=[item for sublist in predictions for item in sublist]
    probs=[item for sublist in predictions_prob for item in sublist]
    labels_flatten= [item for sublist in batch_l for item in sublist]
    f1_sco = metrics.f1_score(labels_flatten,outputs_flatten,average='macro')
    return f1_sco

def train_loop(epochs, model_f, lrate,train_smiles,train_labels, smiles_vl, labels_val, batch_size,device,t_p):
    clip_value=2
    loss_function = nn.CrossEntropyLoss()
    loss_list=[]
    val_loss_list=[]
    data_l=[]
    smiless=[]
    labelss=[]
    patience = 2   
    triggers=0
    previous_val_los=-1

    inputs_tr, masks_tr=enc(train_smiles, t_p)
    inputs_val, masks_val=enc(smiles_vl, t_p)
    dataloader=create_dataloaders(inputs_tr, masks_tr, train_labels, batch_size,device)
    dataloader_vl=create_dataloaders(inputs_val, masks_val, labels_val, batch_size,device)

    optimizer = AdamW(model_f.parameters(),
          lr=lrate,                 eps=1e-8,weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
         num_warmup_steps=0, num_training_steps=total_steps)

    best_epoch = -1  
    for epoch in range(epochs):
        total_loss=0
        print('----------Epoch {}---------'.format(epoch))
        model_f.train()
        for step,batch in enumerate(dataloader):
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            model_f.zero_grad()
            outputs=model_f(batch_inputs, batch_masks)
            loss=loss_function(outputs,batch_labels.type(dtype=torch.long))
            total_loss+=loss.item()
            loss.backward()
            clip_grad_norm_(model_f.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
        avg_tr_loss=total_loss/len(dataloader)
        curr_val_loss = validation(model_f,device,dataloader_vl,loss_function)

        triggers,break_flag,previous_val_los,best_epoch= early_stopping_f1(curr_val_loss,previous_val_los,patience,epoch,best_epoch)#early_stopping(curr_val_loss,previous_val_los,patience,triggers,epoch,best_epoch)#
        if break_flag:  
            break

    return model_f,  best_epoch

def train_loop_simple(epochs, model_f, lrate,smiles,train_labels, batch_size,device,t_p):
    clip_value=2
    loss_function = nn.CrossEntropyLoss()
    loss_list=[]
    val_loss_list=[]
    data_l=[]
    smiless=[]
    labelss=[]

    for sm,l in zip(smiles,train_labels):
        if type(sm) == str:
            smiless.append(sm)
            labelss.append(l)
    inputs_tr, masks_tr=enc(smiless,t_p)
    dataloader=create_dataloaders(inputs_tr, masks_tr, labelss, batch_size,device)
    optimizer = AdamW(model_f.parameters(),
          lr=lrate,                 eps=1e-8,weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
         num_warmup_steps=0, num_training_steps=total_steps)    
    for epoch in range(epochs):
        total_loss=0
        print('----------Epoch {}---------'.format(epoch))
        model_f.train()
        for step,batch in enumerate(dataloader):
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            model_f.zero_grad()
            outputs=model_f(batch_inputs, batch_masks)
            loss=loss_function(outputs,batch_labels.type(dtype=torch.long))
            total_loss+=loss.item()
            loss.backward()
            clip_grad_norm_(model_f.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
        avg_tr_loss=total_loss/len(dataloader)

    return model_f


def evaluation_loop(model_f, test_smiles,test_labels,batch_size,device,t_p):
    test_loss=[]
    total_loss=0
    model_f.eval()
    batch_l=[]
    predictions=[]
    predictions_prob=[]
    loss_function = nn.CrossEntropyLoss()

    inputs_tr, masks_tr=enc(test_smiles,t_p)
    dataloader=create_dataloaders(inputs_tr, masks_tr, test_labels, batch_size,device)

    for step,batch in enumerate(dataloader):
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs=model_f(batch_inputs, batch_masks)
            #print(outputs)
            preds = torch.argmax(outputs, dim=-1)
            preds_prob=[prob.softmax(dim=-1) for prob in outputs]
            predictions.append(preds.tolist())
            predictions_prob.append(preds_prob)
            batch_l.append(batch_labels.tolist())
    outputs_flatten=[item for sublist in predictions for item in sublist]
    probs=[item for sublist in predictions_prob for item in sublist]
    labels_flatten= [item for sublist in batch_l for item in sublist]

    auc = metrics.roc_auc_score(labels_flatten,outputs_flatten, average='macro')
    f1=metrics.f1_score(labels_flatten,outputs_flatten, average='macro')
    precision=metrics.precision_score(labels_flatten,outputs_flatten,average='macro')
    recall=metrics.recall_score(labels_flatten,outputs_flatten,average='macro')
    acc=metrics.accuracy_score(labels_flatten,outputs_flatten)

    f1_binary=metrics.f1_score(labels_flatten,outputs_flatten,pos_label=1, average='binary')
    precision_binary=metrics.precision_score(labels_flatten,outputs_flatten,pos_label=1,average='binary')
    recall_binary=metrics.recall_score(labels_flatten,outputs_flatten,pos_label=1, average='binary')    
    return {'auc':auc, 'f1':f1, 'precision':precision, 'recall':recall, 'accuracy':acc, 'f1_bin':f1_binary, 'pr_bin': precision_binary, 'recall_bin':recall_binary}, labels_flatten, outputs_flatten


if __name__=='__main__':
    seed=3407
    set_seed_all(seed)
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)     
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lrate', type=float, required=True)
    parser.add_argument('--att_heads', type=int, required=True)      
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--augment_numbers', type=int, required=True)
    parser.add_argument('--embs', type=str, required=True)
    parser.add_argument('--ft', type=str, required=True) 
    parser.add_argument('--model_path', type=str, required=True)             
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)


    args= parser.parse_args()
    batch_size=args.batch_size
    epochs=args.epochs
    lrate = args.lrate
    embds=args.embs
    dataset_n=args.dataset
    n_att= args.att_heads
    n_layers = args.layers
    augm_n = args.augment_numbers
    m_path = args.model_path
    t_path = args.tokenizer_path

    current_directory = os.getcwd()
    target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
    target_f = os.path.abspath(os.path.join(target_folder, 'data/benchmark'))
    train_path = os.path.join(target_f, 'train_{}_stratified_0.2_clf.csv'.format( dataset_n))
    test_path = os.path.join(target_f, 'test_{}_stratified_0.2_clf.csv'.format( dataset_n))
    valid_path = os.path.join(target_f, 'validation_{}_stratified_0.2_clf.csv'.format( dataset_n))    
    if args.ft == 'True':
        freeze = True
    else:
        freeze = False
    model=ChemLM_model(dt_n= dataset_n, emb_type = embds, drop_rate=0.1,an=augm_n,tt='clf',n_layers=n_layers,att=n_att,seed_=seed, fr=freeze, model_path=m_path , t_path=t_path)
    model.to(device)

    smiles_tr,labels_tr=load_data(train_path)
    smiles_val,labels_val=load_data(valid_path)
    smiles_test,labels_test=load_data(test_path)
    smiles_tr_val = smiles_tr+smiles_val
    labels_tr_val = labels_tr+labels_val

    model1,  epochs_= train_loop(epochs, model, lrate,smiles_tr,labels_tr, smiles_val, labels_val, batch_size,device, t_p=t_path)
    metrics_cv, l_cv, preds_cv = evaluation_loop(model1, smiles_val,labels_val,batch_size,device,t_p=t_path)
    print(metrics_cv)
    if augm_n==-1:
        suffix = 'trial_ev'
    elif augm_n==0:
        suffix = 'pr_ev'
    else:
        suffix='opt'
    if freeze==True:
        suffix_fr = 'ft'
    else:
        suffix_fr = 'nft'

    model2= train_loop_simple(epochs_, model, lrate,smiles_tr_val,labels_tr_val,  batch_size,device,t_p=t_path)
    metrics_test, l_test, preds_test = evaluation_loop(model2, smiles_test,labels_test,batch_size,device,t_p=t_path)
    print(metrics_test)

    f_results = open('{}/results_{}.txt'.format(save_path, dataset_n),'w')
    for k, v in metrics_test.items():
        f_results.write('{}: {}\n'.format(k,v))
    f_results.close()

    f_p=open('{}/preds_{}.txt'.format(save_path, dataset_n),'w')
    for p in preds_test:
        f_p.write('{}\n'.format(p))
    f_p.close()
    f_l=open('{}/labels_{}.txt'.format(save_path, dataset_n),'w')    
    for l in l_test:
        f_l.write('{}\n'.format(l))
    f_l.close()    
