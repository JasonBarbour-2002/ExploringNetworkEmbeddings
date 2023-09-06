# The code is importing the `seaborn` and `pandas` libraries in Python.
import seaborn as sns
import pandas as pd
import numpy as np
from schulze_voting import *
import matplotlib.pyplot as plt

KLresults = np.load('Evaluate_Embedding/Metrics/KL_div.npy')
ami = np.load('Evaluate_Embedding/OldMetrics/mu/ami.npy')
nmi = np.load('Evaluate_Embedding/OldMetrics/mu/nmi.npy')
ari = np.load('Evaluate_Embedding/OldMetrics/mu/ari.npy')
Micro_F1 = np.load('Evaluate_Embedding/OldMetrics/mu/micro_f1.npy')
Macro_F1 = np.load('Evaluate_Embedding/OldMetrics/mu/macro_f1.npy')

mesure = np.array([ami,nmi,ari,Micro_F1,Macro_F1])
names = np.array(['AMI','NMI','ARI','Micro F1','Macro F1'])
lists = np.array(['Community size distribution', 'Internal degree distribution', 'Internal density', 'Max-ODF', 'Average-ODF', 'Flake-ODF', 'Embededness', 'Internal distance', 'Hub dominance'])
Algos = np.array(['M-GAE','Deepwalk', 'BoostNE','NetMF', 'Walklets','Diff2Vec','M-NMF','Node2Vec','RandNE','LEM'])

RanksMeso = np.zeros((len(lists),len(Algos)))
RanksML = np.zeros((len(names),len(Algos)))
# for mu = 0.1 Meso 
print('\n')
for j in range(KLresults.shape[1]):
    results = KLresults[0,j,:]
    print(results)
    temp = np.argsort(np.abs(results))
    RanksMeso[j] = temp
    print(lists[j])
    print(' & '.join(Algos[temp]))
    print()
# print(RanksMeso)
# for mu = 0.1 ML
for metric in range(mesure.shape[0]):
    M = mesure[metric]
    res = M[0,:,0]
    temp = np.argsort(np.abs(res))
    RanksML[metric] = temp
    print(names[metric])
    print(' & '.join(Algos[temp]))
    print()


Correlation = np.zeros((len(lists),len(names)))
for idx,Metric in enumerate(lists):
    for idy,metric in enumerate(names):
        Correlation[idx,idy] = np.corrcoef(RanksMeso[idx],RanksML[idy])[0,1]

# Heat map
df = pd.DataFrame(Correlation.T,index=names,columns=lists)
sns.heatmap(df,cmap='jet_r',square=True,vmin=-1,vmax=1,annot=True,fmt='.2f')
plt.tight_layout()
plt.xticks(rotation=-45,ha='left')
plt.savefig('Voting/Correlation_0.1.png')
plt.show()

# for mu = 0.7 Meso
for j in range(KLresults.shape[1]):
    results = KLresults[-1,j,:]
    print(results)
    temp = np.argsort(np.abs(results))
    RanksMeso[j] = temp
    print(lists[j])
    print(' & '.join(Algos[temp]))
    print()

# for mu = 0.7 ML
for metric in range(mesure.shape[0]):
    M = mesure[metric]
    res = M[-1,:,0]
    temp = np.argsort(np.abs(res))
    RanksML[metric] = temp
    print(names[metric])
    print(' & '.join(Algos[temp]))
    print()

Correlation = np.zeros((len(lists),len(names)))
for idx,Metric in enumerate(lists):
    for idy,metric in enumerate(names):
        Correlation[idx,idy] = np.corrcoef(RanksMeso[idx],RanksML[idy])[0,1]

# Heat map
df = pd.DataFrame(Correlation.T,index=names,columns=lists)
sns.heatmap(df,cmap='jet_r',square=True,vmin=-1,vmax=1,annot=True,fmt='.2f')
plt.tight_layout()
plt.xticks(rotation=-45,ha='left')
plt.savefig('Voting/Correlation_0.7.png')
plt.show()
