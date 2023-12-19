import deepchem as dc
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,accuracy_score,confusion_matrix
import pandas as pd
from deepchem.models.torch_models import MPNNModel
from deepchem.models import GCNModel,GATModel
import dgl
import torch
import os
import random
from argparse import ArgumentParser
from rdkit import RDLogger 


def set_seed(seed):
  np.random.seed(123)
  tf.random.set_seed(123)
  torch.manual_seed(seed)
  #    torch.manual_seed_all(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  #torch.use_deterministic_algorithms(True)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(dataset):
  print(dataset)
  data=pd.read_csv(dataset,sep='\t')

  smiles=data.loc[:,'smiles'].values.tolist()
  labels=data.loc[:,'y'].values.tolist()
  smiles=[sm for sm in smiles if type(sm)==str]
  labels=[l for sm,l in zip(smiles,labels) if type(sm)==str]
  labels_int=[float(label) for label in labels]
  print('Loaded data')
  return smiles,labels_int

def compute_metrics(preds,labels):
    auc = metrics.roc_auc_score( labels, preds,average='macro')
    f1=metrics.f1_score(labels,preds,average='macro')
    precision=metrics.precision_score(labels,preds,average='macro')
    recall=metrics.recall_score(labels,preds,average='macro')
    acc=metrics.accuracy_score(labels,preds)

    f1_binary=metrics.f1_score(labels,preds, average='binary')
    precision_binary=metrics.precision_score(labels,preds,average='binary')
    recall_binary=metrics.recall_score(labels,preds,average='binary')    
    return {'auc':auc, 'f1':f1, 'precision':precision, 'recall':recall, 'accuracy':acc, 'f1_bin':f1_binary, 'pr_bin': precision_binary, 'recall_bin':recall_binary}

def model_builder_mpnn(**model_params):
  #dt=model_params['dataset']
  dropout=model_params['dropout']
  b=model_params['batch_size']
  lr=model_params['learning_rate']
  model = MPNNModel(mode='classification', n_tasks=1,dropout=dropout,batch_size=b,  learning_rate=lr)#, model_dir='/home/gkallergis/final_results/final_models/mpnn_{}_wval'.format(dt)  )
  return model

def model_builder_gcn(**model_params):
  #dt=model_params['dataset']
  dropout=model_params['dropout']
  b=model_params['batch_size']
  lr=model_params['learning_rate']
  model = GCNModel(mode='classification', n_tasks=1,dropout=dropout, batch_size=b,  learning_rate=lr)#, model_dir='/home/gkallergis/final_results/final_models/gcn_{}_wval'.format(dt) )
  return model

def model_builder_gat(**model_params):
  #dt=model_params['dataset']
  dropout=model_params['dropout']
  b=model_params['batch_size']
  lr=model_params['learning_rate']
  model = GATModel(mode='classification', n_tasks=1,dropout=dropout, batch_size=b,  learning_rate=lr)#, model_dir='/home/gkallergis/final_results/final_models/gat_{}_wval'.format(dt) )
  return model


if __name__=="__main__":
  RDLogger.DisableLog('rdApp.*')
  parser = ArgumentParser()
  parser.add_argument('--save_path', type=str, required=True)
  parser.add_argument('--dataset', type=str, required=True)
  parser.add_argument('--model', type=str, required=True)

  args= parser.parse_args()
  model_name=args.model
  name = args.dataset
  save_path = args.save_path

  set_seed(123)
  params = {
    'learning_rate': [0.00001,0.00005,0.0001,0.0005,0.001],
    'nb_epochs': [20,50,70 ],  
    'dropout':[0.01,0.05,0.1,0.2],
    'batch_size':[8,16,32,64]
      }  
  divisions=['train','validation','test']
  datasets_l=[]
  filename='{}_{}_stratified_0.2_clf.csv'.format('train',name)
  current_directory = os.getcwd()
  target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
  target_f = os.path.abspath(os.path.join(target_folder, 'data/benchmark'))
  train_path = os.path.join(target_f, 'train_{}_stratified_0.2_clf.csv'.format( name))
  test_path = os.path.join(target_f, 'test_{}_stratified_0.2_clf.csv'.format( name))
  if model_name=='gat':
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    model_builder=model_builder_gat

  elif model_name=='gcnn':  
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    model_builder=model_builder_gcn

  elif model_name=='mpnn':
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    model_builder=model_builder_mpnn

  datasets_l=[]
  dgl.use_libxsmm(False)

  divisions=['train','validation','test']
  for i,division in enumerate(divisions):
    filename = os.path.join(target_f, '{}_{}_stratified_0.2_clf.csv'.format(division, name)) 
    smiles,labels=load_data(filename)
    sm_feat_list=[]
    lab_list=[]
    for sm, lab in zip(smiles,labels):
      try:
        sm_feat=featurizer.featurize(sm)
        if sm_feat.size== 0:
          continue
        sm_feat_list.append(sm)
        lab_list.append(lab)
      except:
        continue
    feat=featurizer.featurize(sm_feat_list)
    feat2,label_list2=[],[]
    for f,la in zip(feat,lab_list):
      if not np.any(f):
        feat2.append(f)
        label_list2.append(la)
    lan=np.array(lab_list)

    dataset=dc.data.NumpyDataset(X=feat,y=lan)
    datasets_l.append(dataset)
  optimizer = dc.hyper.GridHyperparamOpt(model_builder)
  metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

  best_model, best_hyperparams, all_results =   optimizer.hyperparam_search(params, datasets_l[0], datasets_l[1], metric=metric)
  print(best_hyperparams)

  f=open('{}/{}_{}_final.txt'.format( save_path,model_name,name),'w')
  f.write('Best Hyper: {}\n'.format(best_hyperparams))
  probs_val_cv=best_model.predict(datasets_l[1])
  arg_vals_cv=np.argmax(probs_val_cv,axis=1)
  metrics_val=compute_metrics(arg_vals_cv,datasets_l[1].y)
  f.write('{}'.format(metrics_val))
  #smiles1,labels1=load_data('/vol/projects/gkallerg/lpmcr/data/datasets/{}_{}_dc_skf.csv'.format(name,'training'))
  smiles1,labels1=load_data('{}_{}_stratified_0.2_clf.csv'.format('train',name))
  smiles2,labels2=load_data('{}_{}_stratified_0.2_clf.csv'.format('validation',name))
  smiles_tr_vl=smiles1+smiles2
  labels_tr_vl=labels1+labels2
  feat_tr_vl= featurizer.featurize(smiles_tr_vl)
  dataset_tr_vl= dc.data.NumpyDataset(X=feat_tr_vl, y=np.array(labels_tr_vl))
  best_model.fit(dataset_tr_vl)
  probs_val=best_model.predict(datasets_l[2])
  arg_vals_test=np.argmax(probs_val,axis=1)
  pred_l = datasets_l[2].y
  f_l = open('{}/labels_{}_{}.txt'.format(save_path,model_name,name),'w')
  for l in pred_l:
    f_l.write('{}\n'.format(l))
  f_l.close()
  f_p = open('{}/preds_{}_{}.txt'.format(save_path,model_name,name),'w')
  for p in arg_vals_test:
    f_p.write('{}\n'.format(p))
  f_p.close()        
  metrics_test=compute_metrics(arg_vals_test,datasets_l[2].y)
  print(metrics_test)
  f.write('{}'.format(metrics_test))
  f.close()
