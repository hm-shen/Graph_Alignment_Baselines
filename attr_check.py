'''
Description: This file check the attributes of two graphs
'''

import cPickle
import numpy as np
import networkx as nx


if __name__ == '__main__':

    # input_path = './data/kdd2018/attributes_data/dblp/dblp950-1/dblp_combined_edges.txt'
    # true_align_path = './data/kdd2018/attributes_data/dblp/dblp950-1/dblp_edges-mapping-permutation.txt'
    #attr_path = './data/kdd2018/attributes_data/dblp/dblp950-1/attributes/attr5-2vals-prob0.000000'
    #attr_path = './data/kdd2018/attributes_data/dblp/dblp950-1/attributes/attr1-29vals-prob0.000000'

    # input_path = './data/kdd2018/dblp/dblp950-1/dblp_combined_edges.txt'
    # true_align_path = './data/kdd2018/dblp/dblp950-1/dblp_edges-mapping-permutation.txt'
    # attr_path = './data/kdd2018/dblp/dblp950-1/attributes/attr1-29vals-prob0.000000'

    input_path = "./data/kdd2018/arenas/arenas990-1/arenas_combined_edges.txt"
    true_align_path = "./data/kdd2018/arenas/arenas990-1/arenas_edges-mapping-permutation.txt"
    attr_path = "./data/kdd2018/arenas/arenas990-1/attributes/attr1-2vals-prob0.050000"

    # load supergraph and true alignments
    super_graph = nx.read_edgelist(input_path, nodetype=int)
    numsuper = nx.number_of_nodes(super_graph)
    true_align = cPickle.load(open(true_align_path, 'rb'))
    attrs = np.load(attr_path)

    assert numsuper % 2 == 0, "ERROR: Number of nodes in supergraph should be even!"

    # create subgraphs
    graph1 = super_graph.subgraph(range(numsuper / 2))
    graph2 = super_graph.subgraph(range(numsuper / 2, numsuper))

    # check true alignments
    permut_index = [true_align[ind] for ind in range(numsuper / 2)]

    adjacency1 = nx.to_scipy_sparse_matrix(graph1)
    adjacency2 = nx.to_scipy_sparse_matrix(graph2)

    print 'Permutation sanity check error is %.1f.'\
        % np.abs(adjacency2[np.ix_(permut_index, permut_index)] - adjacency1).sum()

    print 'Permutation sanity check error is %.1f.'\
        % np.abs(adjacency1[np.ix_(permut_index, permut_index)] - adjacency2).sum()

    N = nx.number_of_nodes(graph1)
    print 'number of nodes in graph is ', N
    attr1 = attrs[0:N,:]
    print 'shape of attr1 is ', attr1.shape
    attr2 = attrs[N:, :]
    print 'shape of attr2 is ', attr2.shape

    score = 0
    for ind in range(N):
        ind_in_graph2 = true_align[ind]
        # print 'corresponding attributes in two graphs are:'
        # print attr1[ind,:]
        # print attr2[ind_in_graph2,:]
        # print 'complete'
        if np.array_equal(attr1[ind,:], attr2[ind_in_graph2,:]):
            # corresponding attributes are the same, GOOD!
            score += 1

    print 'percentage of noises in attributes are:', (1 - float(score) / N)