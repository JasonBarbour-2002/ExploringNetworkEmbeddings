# %%
from ABCD_Graph import ABCD_Graph as ABCD
import scipy.sparse as sp
from tqdm.notebook import tqdm
import numpy as np
import networkx as nx
import warnings
# %%
Current_path = 'Graph_Generation/'
argsABCD = dict(n = 10_000, d_min = 3, d_max = 100, c_min = 10, c_max = 1000,t1=2.7,t2=2.7,mu=0.4)
pathsave = f'{Current_path}Graph_mu/'
# %%
for i in tqdm(np.arange(0.1,0.8,0.1)):
    i = round(i,2)
    argsABCD['mu'] = i
    while True:
        try:
            G = ABCD(**argsABCD,c_max_iter=1000,d_max_iter=1000,path=pathsave+'/useless')
        except Exception as e:
            print(e)
            continue
        if nx.is_connected(G):
            attr = nx.get_node_attributes(G,'community')
            attr = np.array(list(attr. values()))
            np.save(f'{pathsave}Comu_{i}.npy', attr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mat = nx.adjacency_matrix(G)
            sp.save_npz(f'{pathsave}Adj_{i}.npz', mat)
            break

# %%
pathsave = f'{Current_path}Graph_mu_Hyper/'
# %%
for i in tqdm([0.1,0.7]):
    i = round(i,2)
    argsABCD['mu'] = i
    while True:
        try:
            G = ABCD(**argsABCD,c_max_iter=1000,d_max_iter=1000,path=pathsave+'/useless')
        except Exception as e:
            print(e)
            continue
        if nx.is_connected(G):
            attr = nx.get_node_attributes(G,'community')
            attr = np.array(list(attr. values()))
            np.save(f'{pathsave}Comu_{i}.npy', attr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mat = nx.adjacency_matrix(G)
            sp.save_npz(f'{pathsave}Adj_{i}.npz', mat)
            break

# %%