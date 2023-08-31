import numpy as np
from schulze_voting import *
import matplotlib.pyplot as plt

KLresults = np.load('Metrics/KL2_div.npy')
ami = np.load('OldMetrics/mu/ami.npy')
nmi = np.load('OldMetrics/mu/nmi.npy')
ari = np.load('OldMetrics/mu/ari.npy')
Micro_F1 = np.load('OldMetrics/mu/micro_f1.npy')
Macro_F1 = np.load('OldMetrics/mu/macro_f1.npy')

mesure = np.array([ami,nmi,ari,Micro_F1,Macro_F1])
names = np.array(['AMI','NMI','ARI','Micro F1','Macro F1'])
lists = np.array(['Community size distribution', 'Internal degree distribution', 'Internal density', 'Max-ODF', 'Average-ODF', 'Flake-ODF', 'Embededness', 'Internal distance', 'Hub dominance'])
Algos = np.array(['M-GAE','Deepwalk', 'BoostNE','NetMF', 'Walklets','Diff2Vec','M-NMF','Node2Vec','RandNE','LEM'])

allvs = []

for mu in range(KLresults.shape[0]):
    for j in range(KLresults.shape[1]):
        results = KLresults[mu,j,:]
        temp = np.abs(results).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(results))
        v1 = SchulzeVote(ranks,1)
        allvs.append(v1)
allvs2 = []
for mu in range(7):
    for metric in range(5):
        M = mesure[metric]
        res = M[mu,:,0]
        temp = np.argsort(np.abs(res))[::-1]
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(res))
        v1 = SchulzeVote(ranks,1)
        allvs2.append(v1)


RanksMeso = np.zeros((len(lists),len(Algos)))
for idx,Metric in enumerate(lists):
    evaluate = evaluate_schulze(allvs[idx::KLresults.shape[1]],10).candidate_wins
    templist = []
    for i in evaluate:
        templist += i
    RanksMeso[idx] = templist

RanksML = np.zeros((len(names),len(Algos)))
for idx,Metric in enumerate(names):
    evaluate = evaluate_schulze(allvs2[idx::5],10).candidate_wins
    templist = []
    for i in evaluate:
        templist += i
    RanksML[idx] = templist

Correlation = np.zeros((len(lists),len(names)))
for idx,Metric in enumerate(lists):
    for idy,metric in enumerate(names):
        # print(f'{Metric} vs {metric} : ',RanksMeso[idx],RanksML[idy])
        Correlation[idx,idy] = np.corrcoef(RanksMeso[idx],RanksML[idy])[0,1]
        # print(f'{Metric} vs {metric} = {Correlation[idx,idy]}')

# Heat map
import seaborn as sns
import pandas as pd
df = pd.DataFrame(Correlation.T,index=names,columns=lists)
sns.heatmap(df,cmap='jet_r',square=True,vmin=-1,vmax=1,annot=True,fmt='.2f')
plt.tight_layout()
plt.xticks(rotation=-45,ha='left')
plt.savefig('Correlation.png')
plt.show()
