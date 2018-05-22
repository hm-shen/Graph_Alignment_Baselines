'''
Description: This files contains auxiliary functions for baseline tests.
'''

import matlab.engine
import sys
import time
import random
import operator
import cPickle
import logging
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as sps
import networkx as nx

def load_graphs(file_path, true_align_path, attrib_path=None, mode='degree'):

    ''' Loading input supergraph, true alignments and attributes file '''

    logging.debug('Loading input supergraph ...')

    # load supergraph and true alignments
    super_graph = nx.read_edgelist(file_path, nodetype=int)
    numsuper = nx.number_of_nodes(super_graph)
    true_align = cPickle.load(open(true_align_path, 'rb'))

    if (attrib_path is not None and mode == 'attributes'):
        attrs = np.load(attrib_path)
        print 'shape of attrs is:', attrs.shape
        # attrs = cPickle.load(open(attrib_path, 'rb'))
    else:
        attrs = None

    assert numsuper % 2 == 0, "ERROR: Number of nodes in supergraph should be even!"

    # create subgraphs
    graph1 = super_graph.subgraph(range(numsuper / 2))
    graph2 = super_graph.subgraph(range(numsuper / 2, numsuper))

    # check true alignments
    permut_index = [true_align[ind] for ind in range(numsuper / 2)]
    # adjacency1 = sp.sparse.csr_matrix(nx.to_numpy_matrix(graph1))
    # adjacency2 = sp.sparse.csr_matrix(nx.to_numpy_matrix(graph2))

    adjacency1 = nx.to_scipy_sparse_matrix(graph1)
    adjacency2 = nx.to_scipy_sparse_matrix(graph2)

    # print 'Permutation sanity check error is %.1f.'\
    #     % np.abs(adjacency2[np.ix_(permut_index, permut_index)] - adjacency1).sum()

    # print 'Permutation sanity check error is %.1f.'\
    #     % np.abs(adjacency1[np.ix_(permut_index, permut_index)] - adjacency2).sum()


    logging.debug('Input supergraph has been loaded.')

    return graph1, graph2, true_align, attrs

def get_init_align(graph1, graph2, true_align, mode='degree',
                   attrs=None, num_noise=15):

    ''' get init align according to attributes or degree seq '''

    logging.debug('Generating initial alignment matrix using %s ...' % mode)
    print 'Generating initial alignment matrix using %s ...' % mode

    if (mode == 'attributes' and (attrs is not None)):
        # generate initial alignment using attributes
        logging.info('Generate initial alignment matrix using %s ...' % mode)

        return gen_init_align_accord_attrs(attrs)
    else:
        # generate initial alignment using degree
        print 'mode is ', mode
        assert mode == 'degree', "ERROR: Please check initial alignment generation!"

        logging.info('Generate initial alignment matrix using %s ...' % mode)
        N = nx.number_of_nodes(graph1)

        print 'number of noisy edges are: ', num_noise
        return gen_init_align_accord_degree(graph1, graph2, true_align,
                                            # 'degree_only',
                                            'similar_degree',
                                            # 'random_pick_w_true',
                                            num_noise=num_noise)
def get_attr_similarity(array1, array2):

    ''' calculate the similarity between two attribute arrays '''

    array1 = array1.tolist()
    array2 = array2.tolist()

    assert len(array1) == len(array2),\
        "ERROR: the length of two arrays should be the same!"

    score = 0
    for ind in range(len(array1)):
        if array1[ind] == array2[ind]:
            score += 1

    return score

def gen_init_align_accord_similarity(attrs, true_align, pct_init=0.2):

    ''' generate initial alignment matrix accord to attributes similarity '''

    numsuper, _ = attrs.shape

    assert numsuper % 2 == 0, "ERROR: Number of attributes should be even!"

    # get attributes
    attrs1 = attrs[:numsuper/2,:]
    attrs2 = attrs[numsuper/2:,:]

    # construct initial alignment matrix using similarity mat
    num_init = int(np.round(pct_init * (numsuper / 2.0)))

    # construct the similarity matrix first
    simi_mat = np.zeros((numsuper / 2, numsuper / 2))

    print 'Start generating similarity matrix ...'

    for ind in range(numsuper / 2):
        for jnd in range(numsuper / 2):
            simi_mat[ind, jnd] = np.sum((attrs1[ind,:] - attrs2[jnd,:]) == 0)

    rows = []
    cols = []
    maxval = np.max(simi_mat)

    for ind in range(simi_mat.shape[0]):
        # choose the top pct_init as init_align
        simi_ind = simi_mat[ind,:]
        # simi_index = np.nonzero(simi_ind == maxval)[0]
        simi_index = simi_ind.argsort()[::-1][:num_init]
        # simi_index = np.lexsort((np.random.random(simi_ind.size),simi_ind))[-num_init:]
        # for jnd in range(num_init):
        for jnd in range(len(simi_index)):
            rows.append(ind)
            cols.append(simi_index[jnd])
            if not np.array_equal(attrs1[ind,:],attrs2[simi_index[jnd],:]):
                print 'Not match!'

    init_align = sps.csr_matrix((np.ones(len(cols)), (rows, cols)),
                                shape=(numsuper/2, numsuper/2))

    # init_align = init_align.todense()

    # score = 0
    # for key, val in true_align.items():
    #     if  init_align[key,val] == 1:
    #         score += 1
    # print 'pct of correct alignment in init align is:', float(score) / (numsuper / 2)

    print 'Average number of initial alignments for each node is %d'\
                 % np.average(init_align.sum(axis=0))

    logging.info('Average number of initial alignments for each node is %d'\
                 % np.average(init_align.sum(axis=0)))

    logging.debug('Initial alignments are created.')

    # convert to coo format so that it is easier to pass data to matlab
    return init_align.tocoo()


def gen_init_align_accord_attrs(attrs):

    ''' generate initial alignment matrix '''

    logging.info('Generating init-aligns for attributed graph ...')

    numsuper, _ = attrs.shape

    assert numsuper % 2 == 0, "ERROR: Number of attributes should be even!"

    # create buckets to identify nodes with same attrs
    attrslist = map(tuple, attrs) # transform attrs to a list of # tuples
    attrs_bkt1 = {} # hash nodes with same attr to the same bucket
    attrs_bkt2 = {} # hash nodes with same attr to the same bucket
    # save init align
    rows = []
    cols = []

    logging.debug('Hashing nodes with the same attributes to buckets ...')

    for nodeid in range(numsuper / 2):
        if attrs_bkt1.get(attrslist[nodeid]) is not None:
            # attrslist[nodeid] is already in the dict
            attrs_bkt1[attrslist[nodeid]].extend([nodeid])
        else:
            # attrslist[nodeid] is a new attr
            attrs_bkt1[attrslist[nodeid]] = [nodeid]

    for nodeid in range(numsuper / 2, numsuper):
        if attrs_bkt2.get(attrslist[nodeid]) is not None:
            # attrslist[nodeid] is already in the dict
            attrs_bkt2[attrslist[nodeid]].extend([nodeid])
        else:
            # attrslist[nodeid] is a new attr
            attrs_bkt2[attrslist[nodeid]] = [nodeid]

    logging.debug('Hashing complete. Start creating init align ...')

    for key, val in attrs_bkt1.items():
        if key in attrs_bkt2.keys():
            # if two buckets have the same key, nodeid in the val part should
            # be included in init align
            for nodeid in list(val):
                for cand_align in list(attrs_bkt2[key]):
                    rows.append(nodeid)
                    cols.append(cand_align - (numsuper / 2))

    # create init align mat using rows, cols
    init_align = sps.csr_matrix((np.ones(len(cols)), (rows, cols)),
                                shape=(numsuper/2, numsuper/2))

    print 'Average number of initial alignments for each node is %d'\
                 % np.average(init_align.sum(axis=0))

    logging.info('Average number of initial alignments for each node is %d'\
                 % np.average(init_align.sum(axis=0)))

    logging.debug('Initial alignments are created.')

    # convert to coo format so that it is easier to pass data to matlab
    return init_align.tocoo()

def gen_init_align_accord_degree(graph1, graph2, true_align, flag, num_noise):

    ''' generate initial alignment matrix according to flag '''

    if flag == 'degree_only':

        return gen_init_align_accord_degree_only(graph1, graph2)

    elif flag == 'random_pick_w_true':

        return gen_init_align_random_pick_w_true(graph1, graph2, true_align,
                                                 num_noise)

    elif flag == 'similar_degree':

        return gen_init_align_w_similar_degree(graph1, graph2, num_noise)

    else:
        print 'ERROR: Incorrect init alignment generation!'
        sys.exit(-1)

def gen_init_align_w_similar_degree(graph1, graph2, num_noise):

    ''' generate init align using nodes with similar degrees'''
    # def init_align_generator3(graph1, graph2, align_dict, num_noise,
    # seed=0):

    N = nx.number_of_nodes(graph1)

    deg_attrs1 = [len(nbrs) for n,nbrs in graph1.adj.items()]
    deg_attrs2 = [len(nbrs) for n,nbrs in graph2.adj.items()]

    # get dgree dictionary
    dgr_dict1 = dict(zip(range(N), deg_attrs1))
    dgr_dict2 = dict(zip(range(N), deg_attrs2))
    max_dgr = max(max(dgr_dict2.values()), max(dgr_dict1.values()))
    # sort the dictionary by values()
    dgr_arr1 = np.array(sorted(dgr_dict1.items(), key=operator.itemgetter(1)))
    dgr_arr2 = np.array(sorted(dgr_dict2.items(), key=operator.itemgetter(1)))

    # group nodes with the same deg together
    dgr_id = {}
    for deg in range(max_dgr + 1):
        # get node ids with dgree 'deg'
        dgr_id[deg] = [list(dgr_arr1[dgr_arr1[:,1] == deg][:,0]), \
                       list(dgr_arr2[dgr_arr2[:,1] == deg][:,0])]

    # init align
    # init_align = np.zeros((N,N))
    row_total = []
    col_total = []
    for key, val in dgr_id.items():
        if len(val[0]) > 0:
            # print 'current deg is {}'.format(key)
            # val[0] is the node in graph1 has degree 'key'
            # val[1] is the node in graph2 has degree 'key'
            init_deg_key1 = [] # initial alignments for nodes with degree 'key'
            init_deg_key2 = [] # initial alignments for nodes with degree 'key'
            candid_deg = key
            # decrease candid_deg:
            while (len(init_deg_key1) < num_noise) and (candid_deg >= 0):
                if candid_deg in dgr_id:
                    candid_node = dgr_id[candid_deg][1]
                    if len(candid_node) > 0:
                        init_deg_key1 = init_deg_key1 +\
                                    random.sample(candid_node, min(num_noise, len(candid_node)))
                    candid_deg -= 1
                else:
                    candid_deg -= 1
            # print 'init_deg_key1 is ', init_deg_key1
            # increase cand_deg:
            candid_deg = key + 1
            while (len(init_deg_key2) < num_noise) and (candid_deg <= max_dgr):
                if candid_deg in dgr_id:
                    candid_node = dgr_id[candid_deg][1]
                    if len(candid_node) > 0:
                        init_deg_key2 = init_deg_key2 +\
                                    random.sample(candid_node, min(num_noise, len(candid_node)))
                    candid_deg += 1
                else:
                    candid_deg += 1
            # print 'init_deg_key2 is ', init_deg_key2
            # get init_deg_key
            init_deg_key = init_deg_key1 + init_deg_key2
            # set init align for nodes in val[0]
            pos = [(x,y) for x in val[0] for y in init_deg_key]
            rowids, colids = np.transpose(pos)
            # print 'rowids are', rowids
            # print 'colids are', colids
            row_total.extend(rowids)
            col_total.extend(colids)
            # init_align[rowids, colids] = 1
    #return sps.coo_matrix(init_align)
    init_align = sps.csr_matrix((np.ones(len(row_total)), (row_total, col_total)),
                                shape=(N,N))
    print 'Average number of initial alignments for each node is %d'\
                 % np.average(init_align.sum(axis=0))

    return init_align.tocoo()

def gen_init_align_random_pick_w_true(graph1, graph2, true_align, num_noise):

    ''' generate init align using true alignments and random selection '''

    # get total number of nodes in graph
    N = nx.number_of_nodes(graph1)

    # make sure number of elements in dictionary is the same as the
    # number of nodes in graphs
    assert N == len(true_align),\
        "ERROR: size of true_align does not equal to graph size!"
    assert (N - 1) > num_noise, "ERROR: too much noisy alignments!"

    # initialize coordinates of init align matrix
    rows = []
    cols = []

    # create init align
    for nodeid in range(N):
        # firstly add true align to alignment matrix
        rows.append(nodeid)
        cols.append(true_align[nodeid])
        # generate random sample (init align candidates) without replacement
        cand_align = random.sample([idx for idx in range(N) \
                                    if idx != true_align[nodeid]],
                                   num_noise)
        rows.extend([nodeid] * len(cand_align))
        cols.extend(cand_align)

    # create init align mat using rows, cols
    init_align = sps.csr_matrix((np.ones(len(cols)), (rows, cols)),
                                shape=(N, N))

    print 'Average number of initial alignments for each node is %d'\
                 % np.average(init_align.sum(axis=0))

    logging.info('Average number of initial alignments for each node is %d'\
                 % np.average(init_align.sum(axis=0)))

    # convert to coo format so that it is easier for us to pass data to matlab
    return init_align.tocoo()

def gen_init_align_accord_degree_only(graph1, graph2):

    ''' generate initial alignment matrix according to degree '''

    logging.info('Generating initial alignment accord. to degree ...')
    print 'Generating initial alignment accord. to DEGREE ...'

    N1 = nx.number_of_nodes(graph1)
    N2 = nx.number_of_nodes(graph2)

    assert N1 == N2, "Number of nodes in two graphs should be equal"

    # Since we need to (initially) align nodes in two graphs with the
    # same degree together, we can use the degree centrality as an
    # attribute value (attr1 with many different possible vals!)

    # generate degree centrality for each node
    deg_attrs1 = [len(nbrs) for n,nbrs in graph1.adj.items()]
    deg_attrs2 = [len(nbrs) for n,nbrs in graph2.adj.items()]

    # print 'deg_attrs1 has shape: %d, there are %d number of zeros. ' % (len(deg_attrs1), deg_attrs1.count(0))
    # print 'deg_attrs2 has shape: %d, there are %d number of zeros. ' % (len(deg_attrs2), deg_attrs2.count(0))

    # get concatenated attribute vector for degree
    deg_attrs = np.array(deg_attrs1 + deg_attrs2)[:,np.newaxis]

    # get the initial alignment matrix using
    # 'gen_init_align_accord_attrs'

    logging.info('degree-attributes are generated.')

    return gen_init_align_accord_attrs(deg_attrs)


def convert_to_matlab(eng, sparse_mat):

    ''' convert scipy sparse matrix to matlab matrices '''


    data = matlab.double(list(sparse_mat.data))
    # row = matlab.double(list(sparse_mat.row))
    row = eng.plus(matlab.double(list(sparse_mat.row)), matlab.double([1]))
    col = eng.plus(matlab.double(list(sparse_mat.col)), matlab.double([1]))
    # col = matlab.double(list(sparse_mat.col))

    return data, row, col


def run_matlab(graph1, graph2, init_align, configs, path, attribs=None,
               method='netalign'):

    '''
    Run Netalign on the given graph1, graph2, and initial alignment matrix.
    '''

    logging.debug('Running matlab ...')

    # start matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(path['NETALIGN'])

    print 'maximum values in init align row is: ', init_align.row.max()
    print 'maximum values in init align col is: ', init_align.col.max()

    # pass sparse matrix to matlab
    num_nodes1 = nx.number_of_nodes(graph1)
    adjmat1 = nx.to_scipy_sparse_matrix(graph1, format='coo')
    adjmat2 = nx.to_scipy_sparse_matrix(graph2, format='coo')
    adj1_data, adj1_row, adj1_col = convert_to_matlab(eng, adjmat1)
    adj2_data, adj2_row, adj2_col = convert_to_matlab(eng, adjmat2)
    ia_data, ia_row, ia_col = convert_to_matlab(eng, init_align)

    # run netalign

    if method == 'netalign':

        logging.debug('Running %s ...' % method)
        ma, mb, run_time = eng.run_netalign(adj1_row, adj1_col, adj1_data,
                                            adj2_row, adj2_col, adj2_data,
                                            ia_row, ia_col, ia_data,
                                            float(configs['alpha']),
                                            float(configs['beta']),
                                            nargout=3)

        # correct node ids
        ma = np.asarray(ma).flatten() - 1
        mb = np.asarray(mb).flatten() - 1

    elif method == 'klau':

        logging.debug('Running %s ...' % method)
        ma, mb, run_time = eng.run_klau(adj1_row, adj1_col, adj1_data,
                                        adj2_row, adj2_col, adj2_data,
                                        ia_row, ia_col, ia_data,
                                        float(configs['alpha']),
                                        float(configs['beta']),
                                        nargout=3)

        ma = np.asarray(ma).flatten() - 1
        mb = np.asarray(mb).flatten() - 1

    elif method == 'isorank':

        logging.debug('Running %s ...' % method)
        ma, mb, run_time = eng.run_isorank(adj1_row, adj1_col, adj1_data,
                                           adj2_row, adj2_col, adj2_data,
                                           ia_row, ia_col, ia_data,
                                           float(configs['alpha']),
                                           float(configs['beta']),
                                           nargout=3)

        ma = np.asarray(ma).flatten() - 1
        mb = np.asarray(mb).flatten() - 1

    elif method == 'final':

        logging.debug('Running %s ...' % method)

        if attribs is not None:
            # input data contains attributes
            # split the attribs for graph1 and graph2
            print 'Attributes exists! feeding attributes'
            attribs1 = matlab.double(attribs[:num_nodes1,:].tolist())
            attribs2 = matlab.double(attribs[num_nodes1:,:].tolist())
        else:
            # input data does not contain attributes
            attribs1 = matlab.double(None)
            attribs2 = matlab.double(None)

        ma, mb, run_time = eng.run_final(adj1_row, adj1_col, adj1_data,
                                         adj2_row, adj2_col, adj2_data,
                                         ia_row, ia_col, ia_data,
                                         attribs1, attribs2,
                                         float(configs['alpha']),
                                         float(configs['maxiter']),
                                         float(configs['tol']),
                                         nargout=3)

        eng.eval('exception = MException.last;', nargout=0)
        eng.eval('getReport(exception)')

        ma = np.asarray(ma).flatten() - 1
        mb = np.asarray(mb).flatten() - 1
    else:

        logging.error('Invalid method %s !' % method)
        print "ERROR: Invalid method %s" % method
        sys.exit(-1)

    logging.debug('Complete.')

    return dict(zip(mb.tolist(), ma.tolist())), float(run_time)

def cal_accuracy(true_align, est_align):

    '''
    Calculate the accuracy of the alignment results.
    '''

    if len(true_align) == len(est_align):
        print 'true_align and est_align have the same number of elements.'
    else:
        print 'true_align (%d) and est_align (%d) DO NOT have the same number of elements.' \
            % (len(true_align), len(est_align))

    score = 0
    for key, val in true_align.items():
        if est_align.get(key, None) == val:
            # if est_align[key] == val:
            score += 1

    return float(score) / len(true_align)


if __name__ == '__main__':

    '''
    Testing
    '''

    # testing init align generator
    # attrs = np.array([[2,3], [1,4], [2,3], [1,4], [3,7], [10,1],
    #                   [3,7], [2,3], [1,4], [2,3], [1,10], [3,7]])
    # initalign = get_init_align(attrs)
    # print initalign.todense()

    # test run_netalign
    # in_path = './data/kdd2018_old/structure_data/arenas/arenas990-1/arenas_combined_edges.txt'
    # true_align_path = './data/kdd2018_old/structure_data/arenas/arenas990-1/arenas_edges-mapping-permutation.txt'
    # attrs_path =
    # './data/kdd2018_old/structure_data/arenas/arenas990-1/attributes/attr1-2vals-prob0.000000'

    # in_path = './data/kdd2018/arenas/arenas990-1/arenas_combined_edges.txt'
    # true_align_path = './data/kdd2018/arenas/arenas990-1/arenas_edges-mapping-permutation.txt'
    # attrs_path = './data/kdd2018/arenas/arenas990-1/attributes/attr1-2vals-prob0.000000'

    # in_path = './data/kdd2018/PPI/PPI990-1/PPI_combined_edges.txt'
    # true_align_path = './data/kdd2018/structure_data/PPI/PPI990-1/PPI_edges-mapping-permutation.txt'
    # attrs_path = './data/kdd2018/structure_data/PPI/PPI990-1/attributes/attr1-2vals-prob0.000000'

    in_path = './data/kdd2018_old/attributes_data/dblp/dblp950-1/dblp_combined_edges.txt'
    true_align_path = './data/kdd2018_old/attributes_data/dblp/dblp950-1/dblp_edges-mapping-permutation.txt'
    # attrs_path = './data/kdd2018/attributes_data/dblp/dblp950-1/attributes/attr1-29vals-prob0.000000'
    attrs_path = './data/kdd2018_old/attributes_data/dblp/dblp950-1/attributes/attr5-2vals-prob0.000000'


    graph1, graph2, true_align, attrs = load_graphs(in_path,
                                                    true_align_path,
                                                    attrs_path,
                                                    'attributes')

    path = {'NETALIGN': './netalign/matlab/'}

    configs = {'netalign': {'trials': 3, 'alpha': 1.0, 'beta': 1.0},
               'klau': {'trials': 3, 'alpha': 1.0, 'beta': 1.0},
               'isorank': {'trials': 3, 'alpha': 0.5, 'beta': 1.0},
               'final':{'trials': 3, 'maxiter': 40, 'relax': 0, 'alpha': 0.75,
                        'tol': 1e-7}}

    # init_align = get_init_align(graph1, graph2, true_align, mode='attributes', attrs=attrs)
    init_align = gen_init_align_accord_similarity(attrs, true_align)
    # init_align = gen_init_align_accord_attrs(attrs)
    # true_align = {id : id for id in range(n_nodes)}

    for method in ['final', 'netalign']:
        est_align, time = run_matlab(graph1, graph2,
                                     init_align,
                                     configs[method], path,
                                     attribs=attrs,
                                     method=method)
        score = cal_accuracy(true_align, est_align)
        print 'alignment accuracy is %.2f %%' % (score * 100)
