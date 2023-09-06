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
lists = np.array(['Size_Distribution', 'Internal_Deg_dist', 'Intern_Density', 'Max_ODF', 'AverageODF', 'FlakeODF', 'Embededness', 'Internal_Distance', 'Hub_dominance'])
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

allvs = np.array(allvs)
allvs2 = np.array(allvs2)

# for mu<= 0.4 for KL
evaluate = evaluate_schulze(allvs[:5*KLresults.shape[1]],10).candidate_wins
print(f'mu <= 0.4 for KL')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')
print('\n')

# for mu> 0.4 for KL
evaluate = evaluate_schulze(allvs[5*KLresults.shape[1]:],10).candidate_wins
print(f'mu > 0.5 for KL')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')
print('\n')

# for all mu for KL
evaluate = evaluate_schulze(allvs,10).candidate_wins
print(f'All mu for KL')
for i in evaluate:
    str = ', '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')
print('\n')

# for mu<= 0.4 for old metrics
evaluate = evaluate_schulze(allvs2[:5*mesure.shape[0]],10).candidate_wins
print(f'mu <= 0.4 for old metrics')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')
print('\n')
# for mu> 0.4 for old metrics
evaluate = evaluate_schulze(allvs2[5*mesure.shape[0]:],10).candidate_wins
print(f'mu > 0.4 for old metrics')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')
print('\n')
# for all mu for old metrics
evaluate = evaluate_schulze(allvs2,10).candidate_wins
print(f'All mu for old metrics')
for i in evaluate:
    str = ', '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')
print('\n')
