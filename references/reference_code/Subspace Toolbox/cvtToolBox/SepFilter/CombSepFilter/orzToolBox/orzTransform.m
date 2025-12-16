function [V] =  orzTransform(T,X)
% function [V] =  orzTransform(T,X)
%
% T: 変換行列
% X: 変換される行列集合
%
% Y: 変換された行列集合


nSizeX = size(X);
nSizeT = size(T);
nSubNum = prod(nSizeX)/prod(nSizeX(1:2));
X = reshape(X,size(X,1),size(X,2),nSubNum);
V = zeros(nSizeT(1),size(X,2),nSubNum);
for I=1:nSubNum
  V(:,:,I) = T*X(:,:,I);
end
V = reshape(V,[size(V,1),nSizeX(2),nSizeX(3:end),1]);


