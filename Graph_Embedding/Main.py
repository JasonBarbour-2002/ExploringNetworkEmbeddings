import os 
import argparse
import time
import warnings
import numpy as np
import scipy.sparse as sp
import networkx as nx
from utils import Kmeans

Current_path = 'Graph_Embedding'

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
    t1 = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Algo(Graph=Graph , seed=seed,**params)
    t2 = time.process_time()
    t = t2 - t1
    # to save space and because some algorithms return float64, 
    # to level the playing field we convert them to float32
    a = a.astype(np.float32) 
    return a,t

def run(Algo,params,Graph,num_comu,L,i,Thread_id):
    a,t = getEmbed(Algo,params,Graph,Thread_id)
    Label, _ = Kmeans(a,opt = num_comu)
    np.save(f'{Current_path}/Label_{L}/{i}/{Algo.__name__}/{Thread_id}.npy',Label)
    F = open(f'{Current_path}/Label_{L}/{i}/{Algo.__name__}/Times.txt','a')
    F.write(f'{t}\n')
    F.close()

def get_Embeddings(path,Algo,params,Thread_id):
    L = path.split('_')[-1][:-1]
    if L == 'mu':
        R = np.arange(0.1,0.71,0.1)
    else:
        R = np.arange(2,3.1,0.1)
    R = np.round(R,2)
    for i in R:
        print(f'Start with {L} = {i}')
        if not os.path.exists(f'{Current_path}/Label_{L}/{i}/{Algo.__name__}'):
            os.makedirs(f'{Current_path}/Label_{L}/{i}/{Algo.__name__}')
        Graph,attr,num_comu = Load_Graph(path,i)
        run(Algo,params,Graph,num_comu,L,i,Thread_id)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--algo', type=str, default='Deepwalk')
    argparser.add_argument('--Thread', type=str)
    arg = argparser.parse_known_args()[0]
    algo = arg.algo
    Thread_id = int(arg.Thread)
    print('Thread_id: ', Thread_id)
    if algo == 'Deepwalk':
        from Graph_Embedding.embed_utils import Deepwalk as Algo
    elif algo == 'n2v':
        from Graph_Embedding.embed_utils import n2v as Algo
    elif algo == 'Walklet':
        from Graph_Embedding.embed_utils import Walklet as Algo 
    elif algo == 'mnmf':
        from Graph_Embedding.embed_utils import mnmf as Algo
    elif algo == 'boost':
        from Graph_Embedding.embed_utils import boost as Algo
    elif algo == 'diff':
        from Graph_Embedding.embed_utils import diff as Algo
    elif algo == 'LEM':
        from Graph_Embedding.embed_utils import LEM as Algo
    elif algo == 'netmf':
        from Graph_Embedding.embed_utils import netmf as Algo
    elif algo == 'randne':
        from Graph_Embedding.embed_utils import randne as Algo
    elif algo == 'GAE':
        from Graph_Embedding.embed_utils import GAE as Algo
    else:
        raise ValueError('Algorithm not found')
    if algo != 'LEM':
        with open(f'Hyperparameter/Hyperparams_results/{Algo.__name__}.txt') as f:
            f.readline()
            line = f.readline()
            line=line.split('params: ')[1]
            params = eval(line)
    else:
        params = {'dimensions':128, 'maximum_number_of_iterations':100}
    get_Embeddings(f'{Current_path}/Graph_mu/',Algo,params,Thread_id)

