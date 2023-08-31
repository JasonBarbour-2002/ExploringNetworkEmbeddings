import os
import time
import warnings
import numpy as np
import networkx as nx
import scipy.sparse as sp 
from embed_utils import Deepwalk,n2v,Walklet,mnmf,boost,diff,LEM,netmf,randne,GAE
from utils import Kmeans
import multiprocessing as mp

def Load_Graph(path,i):
    arr = sp.load_npz(f'{path}Adj_{i}.npz')
    arr = arr.toarray()
    Graph = nx.from_numpy_array(arr)
    nx.set_node_attributes(Graph, dict(enumerate(np.load(f'{path}Comu_{i}.npy'))), 'community')
    attr = nx.get_node_attributes(Graph,'community')
    attr = np.array(list(attr.values()))
    num_comu = np.unique(attr).shape[0]
    return Graph,attr,num_comu

def getEmbed(Algo,params,Graph,seed):
    t1 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Algo(Graph=Graph , seed=seed,**params)
    t2 = time.time()
    t = t2 - t1
    # to save space and because some algorithms return float64, 
    # to level the playing field we convert them to float32
    a = a.astype(np.float32) 
    return a,t

def Parallel(Algo,params,Times,Graph,num_comu,L,i,cpu):
    for j in range(itr):
        Thread_id = j+itr*cpu
        a,t = getEmbed(Algo,params,Graph,Thread_id)
        Times[f'{Thread_id}'] = t
        Label, _ = Kmeans(a,opt = num_comu)
        np.save(f'Label_{L}/{i}/{Algo.__name__}/{Thread_id}.npy',Label)

def getLabels(path,Algo,params):
    cpu = mp.cpu_count()
    print(f'cpu: {cpu}')
    L = path.split('_')[-1][:-1]
    if L == 'mu':
        R = np.arange(0.1,0.71,0.1)
    else:
        R = np.arange(2,3.1,0.1)
    R = np.round(R,2)
    for i in R:
        print(f'Start with {L} = {i}')
        os.makedirs(f'Label_{L}/{i}/{Algo.__name__}')
        Graph,attr,num_comu = Load_Graph(path,i)
        Times = mp.Manager().dict()
        mp.Pool(cpu,maxtasksperchild=1).starmap(Parallel,[(Algo,params,Times,Graph,num_comu,L,i,cpu) for cpu in range(cpu)])
        T = np.zeros(cpu*itr)
        for k in range(cpu*itr):
            T[k] = Times[f'{k}']
        np.save(f'{L}/{i}/{Algo.__name__}/Times.txt',T)

if __name__ == '__main__':
    itr = 4
    Algo = Deepwalk
    with open(f'results/{Algo.__name__}.txt') as f:
        f.readline()
        line = f.readline()
        line=line.split('params: ')[1]
        params = eval(line)
    print(f'Start with {Algo.__name__}, {params}')
    getLabels('Graph_mu/',Algo,params)