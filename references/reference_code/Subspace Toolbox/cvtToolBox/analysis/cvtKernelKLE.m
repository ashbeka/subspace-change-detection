function [ K,A,D,C ] = cvtKernelKLE( X,dim_1, sigma2)
% [ K,A,D,C ] = cvtKernelKLE( X,dim_1, sigma2) 
% カーネル主成分分析（自己相関基準）
% X:        data
% dim_1:    dimension of subspace
% sigma2:   sigma2
%
% K:        kernem mtrix
% A:        coefficient 
% D:        eigen value
% C:        Contribution rate
%
% It was written by ohkawa on July 7,2009. 
% 
K = cvtKernelMatrix(X,sigma2);
[A,D] = eigs(K,dim_1);
A = A/sqrt(D);
C = sum(diag(D))/trace(K);
D = diag(D)/size(X,2);