'''
Description: This is the baseline test program for Netalign, Klau, FINAL.
'''

import time
import cPickle
import logging
import argparse
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as sps
import networkx as nx
import utility as utl

def arg_parser():

    ''' parse input arguments '''
    # init parser
    parser = argparse.ArgumentParser(description='Run Baselines')

    # add arguments
    parser.add_argument("-p", "--file_path",
                        help="Path to the supergraph",
                        type=str,
                        default='./data/blog995-1/blog_combined_edges.txt')

    parser.add_argument("-t", "--true_alignments",
                        help="Path to true alignments",
                        type=str,
                        default='NONE')

    parser.add_argument("-a", "--attributes",
                        help="Path to attributes",
                        type=str,
                        default='NONE')

    parser.add_argument("-m", "--mode",
                        help="choose mode to generate initial alignment matrix.",
                        type=str,
                        default='degree')

    parser.add_argument("-n", "--name",
                        help="Name of the input data",
                        type=str, default='blog995')

    args = parser.parse_args()

    return args

def main(inargs, configs, path):
    '''
    main()
    '''
    # assert inargs.mode == 'attributes', "ERROR: please feed init align!"

    # read input supergraph
    graph1, graph2, true_align, attrs = utl.load_graphs(inargs.file_path,
                                                        inargs.true_alignments,
                                                        inargs.attributes,
                                                        inargs.mode)

    # create initial alignments using log(n) nodes with similar
    # degrees
    N = nx.number_of_nodes(graph1)
    num_init = int(np.round(np.log2(N)))
    print 'num_init for this graph is %d' % num_init

    # generate initial alignments accord. attributes
    init_align = utl.get_init_align(graph1, graph2, true_align,
                                    inargs.mode, attrs,
                                    num_init)

    # init_align = utl.gen_init_align_accord_similarity(attrs)
    # init_align = utl.gen_init_align_accord_attrs(attrs)

    # run and save results
    results = []
    for method in configs['methods']:
        est_align, total_time = utl.run_matlab(graph1, graph2, init_align,
                                               configs['%s' % method],
                                               path, attribs=attrs,
                                               method=method)

        score = utl.cal_accuracy(true_align, est_align)

        results.append([method, score * 100, float(total_time)])
        print 'method %s has a score of %.2f%%' % (method, score * 100)
        print 'method %s used %.3f seconds.' % (method, total_time)

    results_dataframe = pd.DataFrame(results, columns=['method', 'score', 'time'])
    print "experiment results:\n", results_dataframe

if __name__ == '__main__':

    '''
    set up for logging
    '''
    logging.basicConfig(filename='./logs/baseTests.log', filemode='w',
                        level=logging.DEBUG, format='%(levelname)s %(message)s')

    '''
    set up for numpy precision
    '''
    np.set_printoptions(precision=4)

    logging.debug('Parsing input arguments ...')
    inargs = arg_parser()
    logging.debug('Input arguments have been parsed.')

    '''
    path and configs
    '''
    logging.debug('Loading predefined paths and configs ...')
    path = {'NETALIGN': './netalign/matlab',
            'FINAL': './FINAL/',
            'Klau': './netalign/matlab'}

    configs = {'methods': ['netalign', "klau", "isorank"],
               'netalign': {'alpha': 1, 'beta': 1},
               'klau': {'alpha': 1, 'beta': 1},
               'isorank': {'alpha': 0.5, 'beta': 1},
               'final': {'maxiter': 50.0, 'alpha': 0.9, 'tol': 1e-7}}

    # configs = {'methods': ['isorank'],
    #            'netalign': {'alpha': 1, 'beta': 1},
    #            'klau': {'alpha': 1, 'beta': 1},
    #            'isorank': {'alpha': 0.5, 'beta': 1},
    #            'final': {'maxiter': 50.0, 'alpha': 0.9, 'tol': 1e-7}}


    logging.debug('Paths and configs are loaded.')

    main(inargs, configs, path)
