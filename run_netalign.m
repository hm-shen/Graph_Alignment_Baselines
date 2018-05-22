%% This file receive inputs from python and execute netalign
function [ma, mb, time] = run_netalign(Arow, Acol, Adata,...
                                       Brow, Bcol, Bdata, ...
                                       Lrow, Lcol, Ldata, alpha, beta)
  % add path
  addpath('./netalign/matlab/');

  % get total number of nodes in graph
  N = max(max(Lrow), max(Lcol));

  % construct sparse A,B,L
  % pay attention to the ordering !!!
  g1 = logical(sparse(Arow, Acol, Adata));
  g2 = logical(sparse(Brow, Bcol, Bdata));
  L = sparse(Lrow, Lcol, Ldata, N, N);

  % start experiments
  st_netalign = tic();
  [S,w,li,lj] = netalign_setup(g1,g2,L);
  [x,~,~] = netalignbp(S,w,alpha,beta,li,lj);
  % [ma mb mi overlap weight] = mwmround(x,S,w,li,lj);
  [mb ma mi overlap weight] = mwmround(x,S,w,li,lj);
  time = toc(st_netalign);
  fprintf('Total time used by netalign is %.2f\n', time);
end
