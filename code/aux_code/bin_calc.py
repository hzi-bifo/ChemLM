import pandas as pd
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,accuracy_score,confusion_matrix
from sklearn import metrics
import itertools
import os


#Code for figure 3b, Tables A5, A4
def experimental_files():
	current_directory = os.getcwd()
	target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
	target_f = os.path.abspath(os.path.join(target_folder, 'results/experimental_files'))
	idx_l = [1,2,3,4,5]
	model_names = ['molbert','mpnn', 'gat','gcnn','chemlm' ]
	for model_name in model_names:
		print(model_name)
		for idx in idx_l:
			labels_path = os.path.join(target_f, 'labels_{}_{}_experimental.txt'.format( model_name, idx))
			file1 = open(labels_path,'r')
			# file1 = open('labels_{}.txt'.format( dataset_name),'r')	
			lines_labels = file1.readlines()
			labels_ = [float(l) for l in lines_labels]
			file1.close()

			preds_path = os.path.join(target_f, 'preds_{}_{}_experimental.txt'.format( model_name, idx))
			file2 = open(preds_path,'r')

			#file2 = open('preds_{}.txt'.format( dataset_name),'r')
			lines_preds = file2.readlines()
			preds_ = [float(p) for p in lines_preds]
			file2.close()

			auc = metrics.roc_auc_score( labels_, preds_,average='macro')
			f1=metrics.f1_score(labels_,preds_,average='macro')
			precision=metrics.precision_score(labels_,preds_,average='macro')
			recall=metrics.recall_score(labels_,preds_,average='macro')
			f1_bin=metrics.f1_score(labels_,preds_,pos_label=1,average='binary')
			precision_bin=metrics.precision_score(labels_,preds_,pos_label=1,average='binary')
			recall_bin=metrics.recall_score(labels_,preds_,pos_label=1,average='binary')
			acc=metrics.accuracy_score(labels_,preds_)

			print('General metrics:')
			print('f1:{} \tauc: {}\tPrecision: {}\tRecall: {}\tAccuracy: {}\n'.format(f1,auc, precision,recall,acc))

			print('Binary metrics:')
			print('F1: {}\tPrecision: {}\tRecall: {}\n\n\n'.format(f1_bin, precision_bin,recall_bin))

#Code for Figure 4, Tables A7, A8, A9,10
def benchmark_files(dataset_name):
	current_directory = os.getcwd()
	target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
	target_f = os.path.abspath(os.path.join(target_folder, 'results/benchmark/{}'.format(dataset_name)))

	models = ['gcnn' ]
	for model_name in models:
		print(model_name)
		f_l = 'labels_{}_{}.txt'.format(  model_name, dataset_name)
		labels_path = os.path.join(target_f, f_l)
		file1 = open('{}'.format( labels_path),'r')
		lines_labels = file1.readlines()
		labels_ = [float(l) for l in lines_labels]
		file1.close()
		f_p= 'preds_{}_{}.txt'.format(  model_name, dataset_name)
		preds_path = os.path.join(target_f, f_p)
		file2 = open('{}'.format( preds_path),'r')
		lines_preds = file2.readlines()
		preds_ = [float(p) for p in lines_preds]
		file2.close()

		auc = metrics.roc_auc_score( labels_, preds_,average='macro')
		f1=metrics.f1_score(labels_,preds_,average='macro')
		precision=metrics.precision_score(labels_,preds_,average='macro')
		recall=metrics.recall_score(labels_,preds_,average='macro')
		f1_bin=metrics.f1_score(labels_,preds_,pos_label=1,average='binary')
		precision_bin=metrics.precision_score(labels_,preds_,pos_label=1,average='binary')
		recall_bin=metrics.recall_score(labels_,preds_,pos_label=1,average='binary')
		acc=metrics.accuracy_score(labels_,preds_)

		print('General metrics:')
		print('f1:{} \tauc: {}\tPrecision: {}\tRecall: {}\tAccuracy: {}\n'.format(f1,auc, precision,recall,acc))

		print('Binary metrics:')
		print('F1: {}\tPrecision: {}\tRecall: {}\n\n\n'.format(f1_bin, precision_bin,recall_bin))
		del labels_
		del preds_	

if __name__=='__main__':
	benchmark_files('bbbp')
	benchmark_files('bace')
	benchmark_files('clintox')
	experimental_files()
