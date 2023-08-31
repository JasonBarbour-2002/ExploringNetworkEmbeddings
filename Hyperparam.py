# %%
import os
import argparse
import pickle
import warnings
import numpy as np
import scipy.sparse as sp
import networkx as nx
from embed_utils import Deepwalk,n2v,Walklet,mnmf,boost,diff,netmf,randne,GAE
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.model_selection import ParameterGrid
import multiprocessing as mp
from utils import Kmeans
# %%
def Load_Graph(path,i):
    arr = sp.load_npz(f'{path}Adj_{i}.npz')
    arr = arr.toarray()
    Graph = nx.from_numpy_array(arr)
    nx.set_node_attributes(Graph, dict(enumerate(np.load(f'{path}Comu_{i}.npy'))), 'community')
    attr = nx.get_node_attributes(Graph,'community')
    attr = np.array(list(attr.values()))
    num_comu = np.unique(attr).shape[0]
    return Graph,attr,num_comu
# %%
Graph, attr, num_comu = Load_Graph('Graph_mu_Hyper/',0.1)
Graph2, attr2, num_comu2 = Load_Graph('Graph_mu_Hyper/',0.7)

Algos = [GAE,Deepwalk, boost, netmf, Walklet,diff,mnmf,n2v,randne]
gaeargs = dict (dimensions = [128],
                model = ['linear_vae','linear_ae','gcn_vae','gcn_ae'],
                iterations = [200],
                learning_rate = [0.001,0.01,0.1],
                beta = [0.0],
                lamb = [0.0,0.5,1.0],
                gamma = [0.0,0.5,1.0],
                s_reg = [1,5,10],
                fast_gae = [True]
                )
deepwalkargs = dict (dimensions=[128],
                learning_rate= [0.01,0.05,0.1,0.5,1],
                number_of_negative_samples= [0],
                walk_length= [2,3,4,5],
                walk_number= [100],
                window_size= [5,10,15,20],
                workers=[1])
n2vargs = dict (dimensions=[128],
                learning_rate= [0.01,0.05,0.1],
                number_of_negative_samples= [0],
                walk_length= [2,3,4],
                walk_number= [100],
                window_size=[5,10,15],
                workers= [1],
                pq= [(0.5,1.5),(1.5,0.5),(1,1)])

def special_n2v(g):
    pg = g['pq']
    g['p'] = pg[0]
    g['q'] = pg[1]
    g.pop('pq')
    
mnmfargs = dict (dimensions=[128],
                alpha= [0.01,0.05,0.1],
                beta= [0.01,0.05,0.1],
                clusters= [10,num_comu],
                eta= [1,5,10],
                iterations= [50],
                lambd= [0.01,0.05,0.1])
Walkletsargs = dict (dim_window= [(32,4),(64,2),(16,8),(8,16)],
                    learning_rate= [0.01,0.05,0.1,0.5,1],
                    number_of_negative_samples= [0],
                    walk_length= [2,3,4,5],
                    walk_number= [100],
                    workers=[1])
def special_Walk(g):
    dw = g['dim_window']
    g['dimensions'] = dw[0]
    g['window_size'] = dw[1]
    g.pop('dim_window')
    
boostneargs = dict (alpha= [1,5,10,20],
                    order= [1,2,3],
                    dim_iter= [(32,3),(64,1),(16,7),(8,15)])

def special_boostneargs(g):
    dit = g['dim_iter']
    g['dimensions'] = dit[0]
    g['iterations'] = dit[1]
    g.pop('dim_iter')

diff2vecargs = dict (diffusion_cover= [2,3,4,5],
                diffusion_number= [100],
                dimensions= [128],
                learning_rate= [0.1,0.01,0.05,0.01],
                number_of_negative_samples= [0],
                window_size= [5,10,15,20],
                workers=[1])
netmf_args = dict (dimensions= [128],
                    iteration= [100,200,500],
                    negative_samples= [1,5,10,20],
                    order= [1,2,3])
randne_args = dict (alphas=[[ ]], 
                    dimensions= [128]) 
allAlgos = {GAE:gaeargs,
            Deepwalk:deepwalkargs,
            mnmf:mnmfargs,
            n2v:n2vargs,
            Walklet:Walkletsargs,
            boost:boostneargs,
            diff:diff2vecargs,
            netmf:netmf_args,
            randne:randne_args}
# %%
def Parrallel(Algorithm,grid,results, ids ,special= None, scoring=AMI):
    best_score = -np.inf
    best_params = {}
    for idx, g in enumerate(grid):
        if special is not None:
            special(g)
        print(f'Start:{Algorithm.__name__}, cpu:{ids}, g: {g}')
        if flag:
            if idx < i:
                continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                em = Algorithm(Graph,seed = 42, **g)
                print(f'Finish:{Algorithm.__name__}, cpu:{ids}, em1')
                em2 = Algorithm(Graph2,seed = 42, **g)
                print(f'Finish:{Algorithm.__name__}, cpu:{ids}, em2')
            except Exception as e:
                print(e.with_traceback(None))
                continue
        if np.any(np.isnan(em)):
            score = - np.inf
        elif np.any(np.isinf(em)):
            score = - np.inf
        else :
            Label,_ = Kmeans(em,opt = num_comu)
            Label2,_ = Kmeans(em2,opt = num_comu2)
            score = scoring(attr, Label) + scoring(attr2, Label2)
        if score > best_score:
            best_score = score
            best_params = g
        a = np.where(grid == g)[0]
        print(f'Done: cpu:{ids}, loop{a}, g: {g}')
    results[ids] = np.array([best_score, best_params])

def Tune(Algorithm,grid ,special= None, scoring = AMI):
    grid = np.array(ParameterGrid(grid))
    print(f'Number of Parametere Tuning: {grid.shape}')
    cpu = mp.cpu_count()
    params = np.array_split(grid, cpu)
    print(f'Number of CPU: {cpu}')
    print(f'Number of Parametere Tuning per CPU: {len(params[0])}')
    results = mp.Manager().dict()
    mp.Pool(cpu, maxtasksperchild=1).starmap(Parrallel, [(Algorithm,params[i],results,i,special,scoring) for i in range(cpu)])
    results = np.array(list(results.values()))
    best_score = np.max(results[:,0])
    best_param = results[np.argmax(results[:,0]),1]
    with open(f'./results/{Algorithm.__name__}.txt','w') as f:
        f.write(f'Best score: {best_score}\n')
        f.write(f'Best params: {best_param}\n')
        f.write(f'All results: {results}\n')

def save_checkpoint(state, checkpoint_file):
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(state, f)

def load_checkpoint(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_checkpoint', help='Path to the checkpoint file')
    args, unknown = parser.parse_known_args()
    if args.use_checkpoint:
        print('Loading checkpoint')
        # Load the checkpoint and restore the state
        checkpoint_state = load_checkpoint(args.use_checkpoint)
        Check_A = checkpoint_state['Check_A']
        Alg = [A.__name__ for A in Algos]
        i = Alg.index(Check_A)
        if i == len(Algos)-1:
            exit(0)
        Check_A = Algos[i+1].__name__
    else:
        Check_A = Algos[0].__name__
    flag = False    
    for A in Algos:
        print()
        if A.__name__ == Check_A:
            flag = True
        if not flag:
            continue
        print(f'Tuning {A.__name__}')
        if A.__name__ == n2v.__name__:
            special = special_n2v
        elif A.__name__ == boost.__name__:
            special = special_boostneargs
        elif A.__name__ == Walklet.__name__:
            special = special_Walk
        else:
            special = None
        Tune(A,allAlgos[A],special=special)
        save_checkpoint({'Check_A':A.__name__}, 'Check.chkp')