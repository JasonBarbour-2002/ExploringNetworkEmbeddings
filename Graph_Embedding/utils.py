import numpy as np
import networkx as nx
from sklearn.cluster import KMeans

def Kmeans(Embeddings,opt=None):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,}
    if opt is not None:
        km = KMeans(n_clusters=opt, **kmeans_kwargs).fit(Embeddings)
        return km.labels_,opt

def Middle(b):
    return (b[1:]+b[:-1])/2
def Size_Dist(Labels,**agrs):
    unique_values = np.unique(Labels)
    newlabels = np.searchsorted(unique_values, Labels)
    counts = np.bincount(newlabels)
    maximum = counts.max()+2
    hist, bin_edges = np.histogram(counts, bins=np.arange(0,500))
    hist = hist/hist.sum()
    # hist = np.cumsum(hist)
    return hist, bin_edges

def GetComulist(G,Labels,return_nodes=False,**args):
    Comulist = []
    already = {}
    nodes = list(G.nodes)
    for i,L in enumerate(Labels):
        if L not in already.keys():
            already[L] = len(Comulist)
            Comulist.append([])
        Comulist[already[L]].append(nodes[i])
    Graphs = []
    for Comu in Comulist:
        Graphs.append(G.subgraph(Comu))
    if return_nodes:
        return Graphs,Comulist
    return Graphs
def intra_Deg(G,Labels,Comulist=None):
    if Comulist is None:
        Comulist = GetComulist(G,Labels)
    Intra = []
    for Comu in Comulist:
        Degs = np.array(list(dict(Comu.degree()).values()))
        Intra.append(Degs.mean())
    return Intra
def Internal_Deg_dist(Graph,Labels,Comulist=None,**args):
    INT = np.array(intra_Deg(Graph,Labels,Comulist))
    hist, bin_edges = np.histogram(INT, bins=np.arange(0,9,0.1))
    hist = hist/hist.sum()
    # hist = np.cumsum(hist)
    return hist, bin_edges
def Intern_Density(Graph,Labels,Comulist=None,**args):
    if Comulist is None:
        Comulist = GetComulist(Graph,Labels)
    Intra = []
    for Comu in Comulist:
        Intra.append(nx.density(Comu))   
    Intra = np.array(Intra)
    hist, bin_edges = np.histogram(Intra, bins=np.arange(0,0.6,0.01))
    hist = hist/hist.sum()
    # hist = np.cumsum(hist)
    return hist, bin_edges

def _ODF_helper(Graph,Labels,function,step=1,**args):
    Graphs= GetComulist(Graph,Labels)
    number = len(np.unique(Labels))
    ODF = np.zeros(number)
    for i,comu in enumerate(Graphs):
        if len(comu.nodes) == 1:
            continue
        total_deg = Graph.degree(comu.nodes)
        total_deg = np.array(list(dict(total_deg).values()))
        intra = comu.degree()
        intra = np.array(list(dict(intra).values()))
        external_deg = (total_deg-intra)
        ODF[i] = function(external_deg,total_deg)
    hist, bin_edges = np.histogram(ODF, bins=np.linspace(0,args['max'],step))
    hist = hist/hist.sum()
    # hist = np.cumsum(hist)
    return hist, bin_edges
def Max_ODF(Graph,Labels,**agrs):
    return _ODF_helper(Graph,Labels,lambda x,y: (x.max()/y[x.argmax()]),step=70,max=3)

def AverageOD(Graph,Labels,**agrs):
    return _ODF_helper(Graph,Labels,lambda x,y: (x/y).sum()/y.shape[0],step=70,max=3)

def FlakeODF(Graph,Labels,**args):
    return _ODF_helper(Graph,Labels,lambda x,y: np.where((x - (y-x)) >= 0)[0].shape[0]/y.shape[0],step=70,max=1)

def Embededness(Graph,Labels,**agrs):
    return _ODF_helper(Graph,Labels,lambda x,y: ((y-x)/y).sum()/(y.shape[0]),step=70,max=1)

def InternalDistance(Graph,Labels,Comulist=None,**args):
    if Comulist is None:
        Comulist = GetComulist(Graph,Labels)
    Intra = []
    for Comu in Comulist:
        if nx.is_connected(Comu):
            Intra.append(nx.average_shortest_path_length(Comu))   
        else :
            largestComponents = [Comu.subgraph(c).copy() for c in sorted(nx.connected_components(Comu), key=len, reverse=True)] 
            Intra.append(nx.average_shortest_path_length(largestComponents[0]))
    Intra = np.array(Intra)
    hist, bin_edges = np.histogram(Intra, bins=np.arange(0,7,0.05))
    hist = hist/hist.sum()
    # hist = np.cumsum(hist)
    return hist, bin_edges
    
def Hub_dominance(Graph,Labels,**args):
    return _ODF_helper(Graph,Labels,lambda x,y: ((y-x)).max()/(y.shape[0]-1),step=70,max=1)