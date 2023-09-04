# %%
import numpy as np
from karateclub.neighbourhood import DeepWalk,Node2Vec, Walklets,BoostNE,Diff2Vec,LaplacianEigenmaps,NetMF,RandNE,MNMF
from GAE import GAE as GAE_
# %%
def fitting(Graph,func,**args):
    model = func(**args)
    model.fit(Graph.copy())
    return model.get_embedding()

def Deepwalk(Graph,seed,**args):
    return fitting(Graph,DeepWalk,seed=seed,**args)

def n2v (Graph,seed,**args):
    return fitting(Graph,Node2Vec,seed=seed,**args)

def mnmf(Graph,seed,**args):
    return fitting(Graph,MNMF,seed=seed,**args)

def Walklet(Graph,seed,**args):
    return fitting(Graph,Walklets,seed=seed,**args)

def boost(Graph,seed,**args):
    return fitting(Graph,BoostNE,seed=seed,**args)

def diff(Graph,seed,**args):
    return fitting(Graph,Diff2Vec,seed =seed,**args)

def LEM(Graph,seed,**args):
    return fitting(Graph,LaplacianEigenmaps,seed=seed,**args)
def netmf(Graph,seed,**args):
    return fitting(Graph,NetMF,seed=seed,**args)

def randne(Graph,seed,**args):
    return fitting(Graph,RandNE,seed=seed,**args)

def GAE(Graph,seed,**args):
    return fitting(Graph,GAE_,**args)