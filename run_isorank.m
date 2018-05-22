%% This file receive inputs from python and execute isorank
function [ma, mb, time] = run_isorank(Arow, Acol, Adata, ...
                                      Brow, Bcol, Bdata, ...
                                      Lrow, Lcol, Ldata, ...
                                      alpha, beta)
  % add path
  addpath('./netalign/matlab/');

  % get total number of nodes in graph
  N = max(max(Lrow), max(Lcol));

  % construct sparse A,B,L
  g2 = logical(sparse(Arow, Acol, Adata));
  g1 = logical(sparse(Brow, Bcol, Bdata));
  L = sparse(Lrow, Lcol, Ldata, N, N)';

  % start experiments
  st = tic();
  [S,w,li,lj] = netalign_setup(g1,g2,L);
  [x,~,~] = isorank(S,w,alpha,beta,li,lj);
  [ma mb mi overlap weight] = mwmround(x,S,w,li,lj);
  time = toc(st);
  fprintf('Total time used by isorank is %.2f\n', time);
end
