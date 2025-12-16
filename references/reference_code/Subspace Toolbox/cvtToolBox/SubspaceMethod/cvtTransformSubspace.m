function [V] =  cvtTransformSubspace(T,X)

nSizeX = size(X);
nSizeT = size(T);
nSubNum = prod(nSizeX)/prod(nSizeX(1:2));
X = reshape(X,size(X,1),size(X,2),nSubNum);
V = zeros(nSizeT(1),size(X,2),nSubNum);
for I=1:nSubNum
  V(:,:,I) = orth(T*X(:,:,I));
end
V = reshape(V,[size(V,1),nSizeX(2),nSizeX(3:end),1]);


