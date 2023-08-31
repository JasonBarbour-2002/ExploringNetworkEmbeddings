import numpy as np
from schulze_voting import *
import matplotlib.pyplot as plt

KLresults = np.load('Metrics/KL2_div.npy')
lists = np.array(['Size_Distribution', 'Internal_Deg_dist', 'Intern_Density', 'Max_ODF', 'AverageODF', 'FlakeODF', 'Embededness', 'Internal_Distance', 'Hub_dominance'])
Algos = np.array(['M-GAE','Deepwalk', 'BoostNE','NetMF', 'Walklets','Diff2Vec','M-NMF','Node2Vec','RandNE','LEM'])
allvs = []
allranks = np.zeros((10,7))
for mu in range(KLresults.shape[0]):
    vs = []
    for j in range(KLresults.shape[1]):
        results = KLresults[mu,j,:]
        temp = np.abs(results).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(results))
        v1 = SchulzeVote(ranks,1)
        vs.append(v1)
        allvs.append(v1)
    evaluate = evaluate_schulze(vs,10).candidate_wins
    print(f'mu = {round(0.1+mu*0.1,2)}')
    for idx,i in enumerate(evaluate):
        allranks[i,mu] = idx
        print(Algos[i],end=' ')
    print('\n') 
for idn, a in enumerate(Algos):
    plt.plot(np.arange(0.1,0.8,0.1),allranks[idn],label=a)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
allvs = np.array(allvs)

# For each metric
for idx,i in enumerate(lists):
    evaluate = evaluate_schulze(allvs[idx::KLresults.shape[1]],10).candidate_wins
    print(f'{i} & ',end='')
    for j in evaluate:
        str = ' '.join(Algos[j])
        print(str,end='')
        print(' & '*(Algos[j].shape[0]),end='')
    print('\n')


# For mu < 0.5
evaluate = evaluate_schulze(allvs[:5*KLresults.shape[1]],10).candidate_wins
print(f'mu < 0.5')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')
print('\n')


# For mu >= 0.5
evaluate = evaluate_schulze(allvs[5*KLresults.shape[1]:],10).candidate_wins
print(f'mu >= 0.5')
for i in evaluate:
    str = ' '.join(Algos[i])
    print(str,end= '')
    print(' & '*(Algos[j].shape[0]),end='')
print('\n')
# For all mu
evaluate = evaluate_schulze(allvs,10).candidate_wins
print(f'All mu')
for i in evaluate:
    str = ', '.join(Algos[i])
    print(str,end='')
    print(' & '*(Algos[i].shape[0]),end='')

print('\n') 

