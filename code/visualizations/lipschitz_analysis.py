import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy import stats
import os

#figures of Supplementary Figure 2. Produces the distribution plots for each chemical property. 
def gen_dist(values, types, pr, s_path,t):
	df = pd.DataFrame(
	    {'values': values,
	     'Type': types,
	    })
	g = sns.displot(df,x='values', hue='Type',kde=True,legend=True,palette = [  '#3027D7','#d73027'] )
	g.fig.subplots_adjust(top=.9)
	g.ax.set_title('{}'.format(t), size=18)
	plt.ylabel('Density', fontsize=15)
	plt.xlabel('Ratio', fontsize=15)
	plt.yticks(size=13)
	plt.xticks(size=13)
	plt.tight_layout(w_pad=0.05)
	plt.locator_params(axis='x', nbins=5)
	plt.savefig('{}/dist_{}.pdf'.format(s_path, pr))	

#calculate the p-values for table 2, needs certain version of scipy for alternative arg
def pvals_calc(real_, random_):
	avg_real = np.mean(real_)
	res = stats.ttest_1samp(random_,  avg_real,alternative='greater')
	print(res)
	print(res.confidence_interval(confidence_level=0.95))
	print('\n\n')

#calculate the median values for our and random Lipschitz constant in Table 2
def calc_median(real_dist, random_dist):
	med_real = round(np.median(real_dist),3)
	std_real =round(np.std(real_dist),4)
	med_random = round(np.median(random_dist),3)
	std_random = round(np.std(random_dist),4)
	rat = round(float(med_real)/med_random,3)
	print('& {} (\\pm {}) & {}\t (\\pm {}) & {} \n\n\n'.format(med_real, std_real, med_random, std_random,rat))

#function that processes the files of Lipschitz constants for both ChemLM's and random Lipschitz constant. Produces Table 2 and Suppl Figure 1
def pvals_lip(save_path):
	properties = ['mw', 'qed','psa']
	titles=['Molecular Weight','QED','Polar surface area']
	current_directory = os.getcwd()
	target_folder = os.path.abspath(os.path.join(current_directory, os.pardir, os.pardir))
	target_f = os.path.abspath(os.path.join(target_folder, 'results/lipschitz_distributions/'))

	for p,t in zip(properties,titles):
		print(t)
		file_path = os.path.join(target_f, 'chemlm_random_{}_200n_100rounds_f.txt'.format(p))#'lipschitz_random_{}_200n_100rounds_f.txt'.format(p))		
		f=open(file_path, 'r')
		#f=open( '{}'.format(lipschitz_file), 'r')		
		lines=f.readlines()
		f.close()
		random_k_l, real_k_l, vales, l = [],[],[],[]
		counter=0
		for line in lines:
			vals = line.split('|')
			real_k_l.append(float(vals[1]))
			random_k_l.append(float(vals[0]))
			vales.append(float(vals[0]))	
			vales.append(float(vals[1]))
			r= float(vals[1])
			randk= float(vals[0])
			if randk<=r:
				counter=counter+1
			l.append('random')
			l.append('ChemLM')
		pvals_calc(real_k_l, random_k_l)
		#calc_median(real_k_l, random_k_l)
		#gen_dist(vales, l, p,save_path,t)

if __name__ =='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_path', type=str, required=True)
	args= parser.parse_args()
	save_p = args.save_path
	pvals_lip(save_p)
