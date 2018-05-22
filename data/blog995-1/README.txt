-Adjacency matrix for graph: blog_combined_edges.txt.  This is an edgelist for a supergraph that combines both the original and permuted graph: its adjacency matrix is a block diagonal matrix with the top left block being the original and the bottom right being the permutation.
A = nx.adjacency_matrix(nx.read_edgelist(blog_combined_edges.txt)).todense(), or something like that, will read in this graph.   The original graph is A[:n,:n] and the permuted should be A[n:, n:]

-True alignments: given in the file "blog_edges-mapping-permutation.txt".
Dictionary of the form {u : v} where u,v are in range [1,2,...n], and specifies node u in the original graph's counterpart in graph 2.
cPickle.load(open("blog_edges-mapping-permutation.txt", "rb")) should work

-Attributes: in the attributes/ folder.  These are 2n x d numpy matrices (np.load(ATTRIBUTES_FILENAME) should work), specifying d attribute values for each of the 2n nodes across both graphs (the first n are for the original graph, the second are for the permuted graph).

FINAL can take in attributes.  For NetAlign, IsoRank, and Klau, construct L, the input matrix corresponding to prior info, based on the node attributes.  (That's what we did in HashAlign.)  For now, just construct L such that for each node, its possible alignments are all other nodes that have the exact same values of their attribute(s).  (This should be very helpful when we have a lot of attributes--that's what we found in FINAL, but less so when there are fewer and/or noisy attributes.)

Try running on all the attributes in the different files.  If you were wondering, the format is attrN - Kvals-probP, where N is the number of attributes, K is the number of values each attribute can take on, and P is the probability we flip a (categorical) attribute value.  We want to study how these quantities affect the alignment accuracy.  Alignment should be easier when you have more attributes, more discriminative attributes that can take on more values, and less attribute noise.
