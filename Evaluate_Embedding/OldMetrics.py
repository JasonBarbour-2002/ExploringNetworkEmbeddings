import os
import numpy as np 
from tqdm import tqdm  
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score as AMI, normalized_mutual_info_score as NMI, adjusted_rand_score as ARI, f1_score as F1

Current_path = 'Evaluate_Embedding/'

def Load_Graph(path,i):
    arr = sp.load_npz(f'{path}Adj_{i}.npz')
    arr = arr.toarray()
    Graph = nx.from_numpy_array(arr)
    nx.set_node_attributes(Graph, dict(enumerate(np.load(f'{path}Comu_{i}.npy'))), 'community')
    attr = nx.get_node_attributes(Graph,'community')
    attr = np.array(list(attr.values()))
    num_comu = np.unique(attr).shape[0]
    return Graph,attr,num_comu

def metrics():
    if not os.path.exists(f'{Current_path}OldMetrics'):
        os.makedirs(f'{Current_path}OldMetrics')
    if not os.path.exists(f'{Current_path}OldMetrics/{L}'):
        os.makedirs(f'{Current_path}OldMetrics/{L}')
    ami = np.zeros((len(R),len(Algos),2))
    nmi = np.zeros((len(R),len(Algos),2))
    ari = np.zeros((len(R),len(Algos),2))
    macro_f1 = np.zeros((len(R),len(Algos),2))
    micro_f1 = np.zeros((len(R),len(Algos),2))
    for idx, i in enumerate(R):
        Graph = Graphs[idx]
        attr = attrs[idx]
        num_comu = num_comus[idx]
        for j, algo in tqdm(enumerate(Algos),desc=f'Calculating Metrics for {L} = {i}',total=len(Algos)):
            AMIs = np.zeros(30)
            NMIs = np.zeros(30)
            ARIs = np.zeros(30)
            Macro_F1s = np.zeros(30)
            Micro_F1s = np.zeros(30)
            for k in range (30):
                emb = np.load(f'Label_{L}/{i}/{algo}/{k}.npy')
                AMIs[k] = AMI(attr,emb)
                NMIs[k] = NMI(attr,emb)
                ARIs[k] = ARI(attr,emb)
                Macro_F1s[k] = F1(attr,emb,average='macro')
                Micro_F1s[k] = F1(attr,emb,average='micro')
            ami[idx,j] = np.mean(AMIs), np.std(AMIs)
            nmi[idx,j] = np.mean(NMIs), np.std(NMIs)
            ari[idx,j] = np.mean(ARIs), np.std(ARIs)
            macro_f1[idx,j] = np.mean(Macro_F1s), np.std(Macro_F1s)
            micro_f1[idx,j] = np.mean(Micro_F1s), np.std(Micro_F1s)
    np.save(f'{Current_path}OldMetrics/{L}/ami.npy',ami)
    np.save(f'{Current_path}OldMetrics/{L}/nmi.npy',nmi)
    np.save(f'{Current_path}OldMetrics/{L}/ari.npy',ari)
    np.save(f'{Current_path}OldMetrics/{L}/macro_f1.npy',macro_f1)
    np.save(f'{Current_path}OldMetrics/{L}/micro_f1.npy',micro_f1)


def analise():
    ami = np.load(f'{Current_path}OldMetrics/{L}/ami.npy')
    nmi = np.load(f'{Current_path}OldMetrics/{L}/nmi.npy')
    ari = np.load(f'{Current_path}OldMetrics/{L}/ari.npy')
    macro_f1 = np.load(f'{Current_path}OldMetrics/{L}/macro_f1.npy')
    micro_f1 = np.load(f'{Current_path}OldMetrics/{L}/micro_f1.npy')
    list = [ami,nmi,ari,macro_f1,micro_f1]
    Scores = ['AMI','NMI','ARI','Macro F1','Micro F1']
    for idx,i in enumerate(list):
        for idn, j in enumerate(Algos):
            plt.errorbar(R,i[:,idn,0],yerr=i[:,idn,1],label=j)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel(L)
        plt.ylabel(f'{Scores[idx]} Score')
        plt.title(f'{Scores[idx]} Score for {L}')
        plt.tight_layout()
        plt.savefig(f'{Current_path}OldMetrics/{L}/{Scores[idx]}.png',bbox_inches='tight')
        plt.clf()
Algos = ['M-GAE','Deepwalk', 'BoostNE','NetMF', 'Walklets','Diff2Vec','M-NMF','Node2Vec','RandNE','LEM']
L = 'mu'
R = np.arange(0.1,0.71,0.1)
R = np.round(R,2)
path = f'Graph_Generation/Graph_{L}/'
Graphs = []
attrs = []
num_comus = []
for param in tqdm(R,desc='Loading Graphs'):
    G,A,N = Load_Graph(path,param)
    Graphs.append(G)
    attrs.append(A)
    num_comus.append(N)

metrics()
analise()