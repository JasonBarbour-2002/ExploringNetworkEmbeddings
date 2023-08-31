import numpy as np
from schulze_voting import *
import matplotlib.pyplot as plt
ami = np.load('OldMetrics/mu/ami.npy')
nmi = np.load('OldMetrics/mu/nmi.npy')
ari = np.load('OldMetrics/mu/ari.npy')
Micro_F1 = np.load('OldMetrics/mu/micro_f1.npy')
Macro_F1 = np.load('OldMetrics/mu/macro_f1.npy')

list = [ami,nmi,ari,Micro_F1,Macro_F1]
names = ['AMI','NMI','ARI','Micro F1','Macro F1']
Algos = np.array(['M-GAE','Deepwalk', 'BoostNE','NetMF', 'Walklets','Diff2Vec','M-NMF','Node2Vec','RandNE','LEM'])
allvs = []
allranks = np.zeros((10,7))
for mu in range(7):
    vs = []
    for metric in range(5):
        M = list[metric]
        res = M[mu,:,0]
        temp = np.argsort(np.abs(res))[::-1]
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(res))
        v1 = SchulzeVote(ranks,1)
        vs.append(v1)
        allvs.append(v1)
    evaluate = evaluate_schulze(vs,10).candidate_wins
    print(f'mu = {round(0.1+mu*0.1,2)}')
    for idx,i in enumerate(evaluate):
        allranks[i,mu] = idx
        print(Algos[i],end=' ')
    print('\n')

allvs = np.array(allvs)

# For each metric
for idx,i in enumerate(names):
    evaluate = evaluate_schulze(allvs[idx::5],10).candidate_wins
    print(f'{i} & ',end='')
    for j in evaluate:
        str = ' '.join(Algos[j])
        print(str,end='')
        print(' & '*(Algos[j].shape[0]),end='')
    print('\n')
# For mu < 0.5
evaluate = evaluate_schulze(allvs[:25],10).candidate_wins
print(f'mu < 0.5')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,' & ', end = '')
print('\n')

# for mu >= 0.5
evaluate = evaluate_schulze(allvs[25:],10).candidate_wins
print(f'mu >= 0.5')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,' & ',end='')
print('\n')



# For all mu 
evaluate = evaluate_schulze(allvs,10).candidate_wins
print(f'All mu')

for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,' & ',end='')
print('\n')