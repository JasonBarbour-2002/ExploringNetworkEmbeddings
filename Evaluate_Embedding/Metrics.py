import os 
from tqdm import tqdm as tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.special import rel_entr
from utils import Size_Dist,Internal_Deg_dist,Intern_Density,Max_ODF,Middle,AverageOD,FlakeODF,Embededness,InternalDistance,Hub_dominance

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

def Just_Graph():
    if not os.path.exists(f'{Current_path}Metrics'):
        os.makedirs(f'{Current_path}Metrics')
    # Just the Graphs without embedding
    if not os.path.exists(f'{Current_path}Metrics/Graphs'):
        os.makedirs(f'{Current_path}Metrics/Graphs')
    for id,M in enumerate(Metrics):
        for idx,param in enumerate(R):
            Graph = Graphs[idx]
            attr = attrs[idx]
            num_comu = num_comus[idx]
            h,b = M(Graph = Graph,Labels = attr,size= num_comu)
            b = Middle(b)
            plt.plot(b,h,'o-',label=f'{L} = {param}')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel(Metric_name[id])
        plt.ylabel('cumulative distribution')
        plt.tight_layout()
        plt.savefig(f'Evaluate_Embedding/Metrics/Graphs/{Metric_name[id]}.png')
        plt.close()

# Graph and embedding
def Graph_Embedding():
    if not os.path.exists(f'{Current_path}/Metrics/Embedding'):
        os.makedirs(f'{Current_path}Metrics/Embedding')
    for id,M in enumerate(Metrics):
        if not os.path.exists(f'{Current_path}Metrics/Embedding/{Metric_name[id]}'):
            os.makedirs(f'{Current_path}Metrics/Embedding/{Metric_name[id]}')
        if not os.path.exists(f'{Current_path}Metrics/Saved/{Metric_name[id]}'):
            os.makedirs(f'{Current_path}Metrics/Saved/{Metric_name[id]}')
        for idx,param in enumerate(R):
            Graph = Graphs[idx]
            attr = attrs[idx]
            num_comu = num_comus[idx]
            h,b = M(Graph = Graph,Labels = attr,size= num_comu)
            b = Middle(b)
            for Algo in tqdm(Algos, desc=f'{Metric_name[id]} for {L} = {param}'):
                h2 = np.zeros((30,h.shape[0]))
                for i in range(30):
                    try:
                        emb = np.load(f'Graph_Embedding/Label_{L}/{param}/{Algo}/{i}.npy')
                        h2[i],_ = M(Graph = Graph,Labels = emb,size= num_comu)
                        np.save(f'Evaluate_Embedding/Metrics/Saved/{Metric_name[id]}/{L}_{param}_{Algo}_{i}.npy',h2[i])
                    except:
                        print(f'error in loading embedding file {i} for {Algo}')
                        pass
                h_res = np.mean(h2,axis=0)
                err = np.std(h2,axis=0)
                plt.errorbar(b,h_res,err,fmt='o-',label=f'{Algo}')
            plt.plot(b,h,'k--',label=f'{L} = {param}',zorder=100)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.xlabel(Metric_name[id])
            plt.ylabel('cumulative distribution')
            plt.tight_layout()
            plt.savefig(f'{Current_path}Metrics/Embedding/{Metric_name[id]}/{L}_{param}.png')
            plt.close()

def Times():
    Algo_names = np.array(Algos)
    fig, ax = plt.subplots()
    for i,Algo in enumerate(Algos):
        Times = np.zeros((len(R),30))
        for idx,param in enumerate(R):
                time = np.loadtxt(f'Graph_Embedding/Label_{L}/{param}/{Algo}/Times.txt')
                Times[idx] = time[:30]
        Times = Times/60
        Mean = np.mean(Times)
        std = np.std(Times)
        ax.bar(i,Mean,0.5,yerr=std,label=f'{Algo}',capsize=2)
    ax.set_xticks(np.arange(len(Algos)),Algo_names,rotation=90)
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Time (min)')
    plt.tight_layout()
    plt.savefig(f'{Current_path}Metrics/Times.png')
    plt.close()
    # plt.show()

def Ks_div():
    KL_div = np.zeros((len(R),len(Metrics),len(Algos)))
    for idx, param in tqdm(enumerate(R),total=len(R),desc='Calculating KS'):
        for i,M in enumerate(Metrics):
            Graph = Graphs[idx]
            attr = attrs[idx]
            num_comu = num_comus[idx]
            h,b = M(Graph = Graph,Labels = attr,size= num_comu)
            b = Middle(b)
            h[h ==0] = np.finfo(float).eps
            for j,Algo in enumerate(Algos):
                h2 = np.zeros((30,h.shape[0]))
                for k in range(30):
                    try:
                        h2[k] = np.load(f'Evaluate_Embedding/Metrics/Saved/{Metric_name[i]}/{L}_{param}_{Algo}_{k}.npy')
                    except:
                        print(f'error in loading embedding file {k} for {Algo}')
                        pass
                h_res = np.mean(h2,axis=0)
                h_res[h_res == 0] = np.finfo(float).eps
                KL_div[idx,i,j] =  np.sum(rel_entr(h_res,h))   
    np.save(f'{Current_path}Metrics/KL_div.npy',KL_div)

def Analise():
    KL = np.load(f'{Current_path}Metrics/KL_div.npy')
    if not os.path.exists(f'{Current_path}Metrics/Analise/KL'):
        os.makedirs(f'{Current_path}Metrics/Analise/KL')
    for i,M in enumerate(Metrics):
        plt.rcParams['text.usetex'] = True
        for j,Algo in enumerate(Algos):
            plt.plot(R,KL[:,i,j],'o-',label=f'{Algo}')
        plt.xlabel(r'$\mu$',fontsize=30)
        plt.ylabel('KL',fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
        plt.savefig(f'{Current_path}Metrics/Analise/KL/{Metric_name[i]}.pdf',dpi=300)
        plt.close()
Metrics = [Size_Dist,Internal_Deg_dist,Intern_Density,Max_ODF,AverageOD,FlakeODF,Embededness,InternalDistance,Hub_dominance]
Metric_name = ['Size Distribution','Internal Degree distribution','Internal Density','Max-ODF','Average-ODF','Falke-ODF','Embededness','Internal Distance','Hub dominance']

Algos = ['M-GAE','DeepWalk','BoostNE','NetMF', 'Walklets','Diff2Vec','M-NMF','Node2Vec','RandNE','LEM']
L = 'mu'
R = np.arange(0.1,0.71,0.1)
R = np.round(R,2)
path = f'Graph_Generation/Graph_mu/'
Graphs = []
attrs = []
num_comus = []
for param in tqdm(R,desc='Loading Graphs'):
    G,A,N = Load_Graph(path,param)
    Graphs.append(G)
    attrs.append(A)
    num_comus.append(N)
# Just_Graph()
# Graph_Embedding()
# Times()
Ks_div()
Analise()