function [V D] =  cvtBasisVectorSVD(X,nSubDim)
% [V D] =  cvtBasisVectorSVD(X,nSubDim)
% This function calculates the orthonormal basis vector of subspace.
%
% ----INPUT----
% X:Data
% nSubDim?FDimension of Subspace
% ----OUTPUT----
% V:Orthonormal Basis Vectors
% D:Eigen Values
%
% It was written by Yasuhiro Ohkawa.
%
nSizeX = size(X);
nSetNum = prod(nSizeX)/prod(nSizeX(1:2));
X = reshape(X,size(X,1),size(X,2),nSetNum);
V = zeros(size(X,1),nSubDim,nSetNum);
D = zeros(nSubDim,nSetNum);
for I=1:nSetNum
    [V(:,:,I) S] = svds(X(:,:,I),nSubDim);
    D(:,I) = (diag(S).^2)/size(S,2);
end
V = reshape(V,[nSizeX(1),nSubDim,nSizeX(3:end),1]);
D = reshape(D,[nSubDim,nSizeX(3:end),1]);




