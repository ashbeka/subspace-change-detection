function [A,D,K ] = sbsKernelBasisVector( X,nSubDim, nSigma2)
% [A,D,K] =  sbsKernelBasisVector(X,nSubDim)
% This function calculates the orthonormal basis vector of subspace in Feature space.
%
% ----INPUT----
% X:Data
% nSubDimÅFDimension of Subspace
% nSigma2 : Parametor of Gaussian Kernel
% ----OUTPUT----
% A: Combination coefficient
% D: Eigen Values
% K: Kernel Matrix(Gram Matrix)
%
% It was written by Yasuhiro Ohkawa.
%

OPTS.disp = 0;
X = X(:,:);

K = sbsKernelMatrix(X,nSigma2);
% A:Combination coefficient
[A,tmp] = eigs(K,nSubDim,'lm',OPTS);
t_eval = diag(tmp);
%  Normalizing all basis-vectors
for E=1:nSubDim,
    A(:,E) = A(:,E)/(sqrt(t_eval(E)));
end;
% Eigen value
D = t_eval/size(X,2);
