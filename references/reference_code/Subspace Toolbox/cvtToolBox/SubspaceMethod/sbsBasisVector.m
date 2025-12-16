function [V D] =  sbsBasisVector(X,nSubDim)
% [V D] =  sbsBasisVector(X,nSubDim)
% This function calculates the orthonormal basis vector of subspace.
%
% ----INPUT----
% X:Data
% nSubDimÅFDimension of Subspace
% ----OUTPUT----
% V:Orthonormal Basis Vectors
% D:Eigen Values
%
% It was written by Yasuhiro Ohkawa.
%

OPTS.disp = 0;

X = X(:,:);
R = X*X'/size(X,2);
[V tmpD] = eigs(R,nSubDim,'lm',OPTS);
D = diag(tmpD);