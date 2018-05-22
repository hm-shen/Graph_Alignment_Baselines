import cPickle
import numpy as np
import networkx as nx

aligns = cPickle.load(open("arenas_edges-mapping-permutation.txt", "rb"))
Gcomb = nx.read_edgelist("arenas_combined_edges.txt", nodetype=int)
Acomb = nx.to_numpy_matrix(Gcomb)
n = Gcomb.number_of_nodes()/2
A1 = Acomb[:n,:n]
A2 = Acomb[n:,n:]
P = np.zeros((n,n))
P[aligns.values(), aligns.keys()] = 1
error = np.count_nonzero(P.dot(A1).dot(P.T) - A2)
# for node in aligns.keys():
#   P[aligns[node], node] = 1
# error = np.count_nonzero(np.dot( np.dot(P, A1), P.T) - A2)

print "Error: ", error
