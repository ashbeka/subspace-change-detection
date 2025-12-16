function [A,D,K,C] =  cvtKernelBasisVector(X,nSubDim,Sigma2)
%[A K C D] =  cvtKernelBasisVector(X,nSubDim,Sigma2)
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
% %
% 
% OPTS.disp = 0;
% X = X(:,:);
% 
% K = sbsKernelMatrix(X, Sigma2);
% % A:Combination coefficient
% [A,tmp] = eigs(K,nSubDim,'lm',OPTS);
% t_eval = diag(tmp);
% %  Normalizing all basis-vectors
% for E=1:nSubDim,
%     A(:,E) = A(:,E)/(sqrt(t_eval(E)));
% end;
% % Eigen value
% D = t_eval/size(X,2);


nSizeX = size(X);
nSubNum = prod(nSizeX)/prod(nSizeX(1:2));
X = reshape(X,size(X,1),size(X,2),nSubNum);

K = zeros(size(X,2),size(X,2),nSubNum);
A = zeros(size(X,2),nSubDim,nSubNum);
C = zeros(nSubNum,1);
D = zeros(nSubDim,nSubNum);
for I=1:nSubNum
    [ K(:,:,I),A(:,:,I),C(I),D(:,I) ] = cvtKernelPCA(X(:,:,I),nSubDim,Sigma2);
end

K = reshape(K,[nSizeX(2),nSizeX(2),nSizeX(3:end),1]);
A = reshape(A,[nSizeX(2),nSubDim,nSizeX(3:end),1]);
if(~isempty(nSizeX(3:end)))
    C = reshape(C,[nSizeX(3:end),1])';
end
D = reshape(D,[nSubDim,nSizeX(3:end),1]);

