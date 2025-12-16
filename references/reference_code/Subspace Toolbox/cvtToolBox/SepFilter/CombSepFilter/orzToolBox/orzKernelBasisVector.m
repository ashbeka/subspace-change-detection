function [A,D,C,K] = orzKernelBasisVector(X,nSubDim,nSigma)
%function [A,D,C,K] = orzKernelBasisVector(X,nSubDim,nSigma)


nSizeX = size(X);
nSubNum = prod(nSizeX)/prod(nSizeX(1:2));
X = reshape(X,size(X,1),size(X,2),nSubNum);

A = zeros(size(X,2),nSubDim,nSubNum);
D = zeros(nSubDim,nSubNum);
C = zeros(nSubNum,1);
K = zeros(size(X,2),size(X,2),nSubNum);
for I=1:nSubNum
    [A(:,:,I),D(:,I),C(I),K(:,:,I)] = orzKPCA(X(:,:,I),nSubDim,nSigma,'R');
end

if size(X,3) ~= 1
    A = reshape(A,[nSizeX(2),nSubDim,nSizeX(3:end),1]);
    D = reshape(D,[nSubDim,nSizeX(3:end),1]);
    C = reshape(C,[nSizeX(3:end),1])';
    K = reshape(K,[nSizeX(2),nSizeX(2),nSizeX(3:end),1]);    
end



