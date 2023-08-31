import os 
from tqdm import tqdm as tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.stats as stats
from scipy.special import rel_entr
from utils import Size_Dist,Internal_Deg_dist,Intern_Density,Max_ODF,Middle,AverageOD,FlakeODF,Embededness,InternalDistance,Hub_dominance

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
    if not os.path.exists('Metrics'):
        os.makedirs('Metrics')
    # Just the Graphs without embedding
    if not os.path.exists('Metrics/Graphs'):
        os.makedirs('Metrics/Graphs')
    for M in Metrics:
        for idx,param in enumerate(R):
            Graph = Graphs[idx]
            attr = attrs[idx]
            num_comu = num_comus[idx]
            h,b = M(Graph = Graph,Labels = attr,size= num_comu)
            b = Middle(b)
            plt.plot(b,h,'o-',label=f'{L} = {param}')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel(M.__name__)
        plt.ylabel('cumulative distribution')
        plt.tight_layout()
        plt.savefig(f'Metrics/Graphs/{M.__name__}')
        plt.close()

# Graph and embedding
def Graph_Embedding():
    if not os.path.exists('Metrics/Embedding'):
        os.makedirs('Metrics/Embedding')
    for M in Metrics:
        if not os.path.exists(f'Metrics/Embedding/{M.__name__}'):
            os.makedirs(f'Metrics/Embedding/{M.__name__}')
        for idx,param in enumerate(R):
            Graph = Graphs[idx]
            attr = attrs[idx]
            num_comu = num_comus[idx]
            h,b = M(Graph = Graph,Labels = attr,size= num_comu)
            b = Middle(b)
            for Algo in tqdm(Algos, desc=f'{M.__name__} for {L} = {param}'):
                h2 = np.zeros((30,h.shape[0]))
                for i in range(30):
                    try:
                        emb = np.load(f'Label_{L}/{param}/{Algo}/{i}.npy')
                        h2[i],_ = M(Graph = Graph,Labels = emb,size= num_comu)
                    except:
                        print(f'error in loading embedding file {i} for {Algo}')
                        pass
                h_res = np.mean(h2,axis=0)
                err = np.std(h2,axis=0)
                plt.errorbar(b,h_res,err,fmt='o-',label=f'{Algo}')
            plt.plot(b,h,'k--',label=f'{L} = {param}',zorder=100)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.xlabel(M.__name__)
            plt.ylabel('cumulative distribution')
            plt.tight_layout()
            plt.savefig(f'Metrics/Embedding/{M.__name__}/{L}_{param}.png')
            plt.close()

# Time For each algorithm
# Algos names
def Times():
    Algo_names = np.array(Algos)
    fig, ax = plt.subplots()
    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    FullTime = 0
    for i,Algo in enumerate(Algos):
        # if Algo == netmf:
        #     continue
        Times = np.zeros((len(R),30))
        for idx,param in enumerate(R):
                time = np.loadtxt(f'Label_{L}/{param}/{Algo}/Times.txt')
                Times[idx] = time[:30]
        Times = Times/60
        Mean = np.mean(Times)
        std = np.std(Times)
        ax.bar(i,Mean,0.5,yerr=std,label=f'{Algo}',capsize=2)
        # axins.bar(i,Mean,0.5,yerr=std,capsize=2)
        if Algo == 'NetMF':
            continue
        FullTime += np.sum(Times)
    print(FullTime/60/24)
    ax.set_xticks(np.arange(len(Algos)),Algo_names,rotation=90)
    # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Time (min)')
    # axins.set_xlim(-1, len(Algos))
    # axins.set_ylim(0, 9)
    # axins.set_xticks(np.arange(len(Algos)),Algo_names,rotation=90)
    # axins.set_yticks(np.arange(0, 9, 2))
    # ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f'Metrics/Times.png')
    plt.show()

def Ks_div():
    KS_div = np.zeros((len(R),len(Metrics),len(Algos)))
    KL_div = np.zeros((len(R),len(Metrics),len(Algos)))
    LSE = np.zeros((len(R),len(Metrics),len(Algos)))
    KL2_div = np.zeros((len(R),len(Metrics),len(Algos)))
    for idx, param in tqdm(enumerate(R),total=len(R),desc='Calculating KS'):
        for i,M in enumerate(Metrics):
            Graph = Graphs[idx]
            attr = attrs[idx]
            num_comu = num_comus[idx]
            h,b = M(Graph = Graph,Labels = attr,size= num_comu)
            b = Middle(b)
            h[h ==0] = np.finfo(float).eps
            for j,Algo in tqdm(enumerate(Algos),desc=f'{M.__name__} for {L} = {param}',total=len(Algos)):
                h2 = np.zeros((30,h.shape[0]))
                for k in range(30):
                    try:
                        emb = np.load(f'Label_{L}/{param}/{Algo}/{k}.npy')
                        h2[k],_ = M(Graph = Graph,Labels = emb,size= num_comu)
                    except:
                        print(f'error in loading embedding file {k} for {Algo}')
                        pass
                h_res = np.mean(h2,axis=0)
                h_res[h_res == 0] = np.finfo(float).eps
                KL2_div[idx,i,j] =  np.sum(rel_entr(h_res,h))   
                KS_div[idx,i,j] = stats.ks_2samp(h,h_res)[1]
                KL_div[idx,i,j] = np.sum(rel_entr(h,h_res))   
                LSE[idx,i,j] = np.sqrt(np.sum((h-h_res)**2)/h.shape[0])
    np.save('Metrics/KS_div.npy',KS_div)
    np.save('Metrics/KL_div.npy',KL_div)
    np.save('Metrics/KL2_div.npy',KL2_div)
    np.save('Metrics/LSE.npy',LSE)

def Analise():
    KL = np.load('Metrics/KL_div.npy')
    KL2 = np.load('Metrics/KL2_div.npy')
    KS = np.load('Metrics/KS_div.npy')
    LSE = np.load('Metrics/LSE.npy')
    list = [KL]
    list2 = ['KL']
    for idx, l in enumerate(list):
        if not os.path.exists(f'Metrics/Analise/{list2[idx]}'):
            os.makedirs(f'Metrics/Analise/{list2[idx]}')
        for i,M in enumerate(Metrics):
            plt.rcParams['text.usetex'] = True
            for j,Algo in enumerate(Algos):
                plt.plot(R,l[:,i,j],'o-',label=f'{Algo}')
            if idx == 3:
                plt.yscale('log')
            # if M in [Max_ODF,Intern_Density,FlakeODF]:
            #     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
            plt.xlabel(r'$\mu$',fontsize=30)
            plt.ylabel(list2[idx],fontsize=30)
            plt.xticks(fontsize=30)
            # if M == InternalDistance:
            #     plt.yticks(np.arange(-10.0,8,4),['$-10$', '$-6$', '$-2$', '$2$', '$6$'],fontsize=30)
            # else:
            plt.yticks(fontsize=30)
            # plt.title(f'{Metric_name[i]}')
            plt.tight_layout()
            plt.savefig(f'Metrics/Analise/{list2[idx]}/{M.__name__}.pdf',dpi=300)
            plt.close()
Metrics = [Size_Dist,Internal_Deg_dist,Intern_Density,Max_ODF,AverageOD,FlakeODF,Embededness,InternalDistance,Hub_dominance]
Metric_name = ['Size Distribution','Internal Degree distribution','Internal Density','Max-ODF','Average-ODF','Falke-ODF','Embededness','Internal Distance','Hub dominance']
def randne3():
    pass
Algos = ['M-GAE','DeepWalk','BoostNE','NetMF', 'Walklets','Diff2Vec','M-NMF','Node2Vec','RandNE','LEM']
L = 'mu'
R = np.arange(0.1,0.71,0.1)
R = np.round(R,2)
path = f'Graph_{L}/'
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