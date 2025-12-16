function [V D C] = orzBasisVector(X,nSubDim)
%function [V D] =  orzBasisVector(X,nSubDim)
% X:        入力行列
% nSubDim:  部分空間の次元
%
% V:        部分空間の基底ベクトル
% D:        固有値
% C:        寄与率
%


OPTS.disp = 0;
nSizeX = size(X);
nSubNum = prod(nSizeX)/prod(nSizeX(1:2));
X = reshape(X,size(X,1),size(X,2),nSubNum);
%X = orzNormalize(X);

V = zeros(size(X,1),nSubDim,nSubNum);
D = zeros(nSubDim,nSubNum);
C = zeros(1,nSubNum);
for I=1:nSubNum
    [Z,V(:,:,I),D(:,I),C(I)] = orzPCA(X(:,:,I),nSubDim,'R');    
end
V = reshape(V,[nSizeX(1),nSubDim,nSizeX(3:end),1]);
D = reshape(D,[nSubDim,nSizeX(3:end),1]);
C = reshape(C,[nSizeX(3:end),1]);



