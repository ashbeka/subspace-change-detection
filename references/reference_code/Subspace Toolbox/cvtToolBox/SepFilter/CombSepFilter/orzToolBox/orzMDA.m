function [Z,T,J,N] = orzMDA(X,L,cRate)
%function [Y,V,J,M] = orzMDA(X,L,cRate)
% èdîªï ï™êÕ ver.1.00 by ohkawa
% input
%  X: matrix of (dimÅ~m), column vectors
%  L: vector of m-dimension, L(i) is label of X(:,i).
% output
%  Y: îªï ãÛä‘Ç÷éÀâeÇ≥ÇÍÇΩX
%  V: îªï é≤
%  J: ï™ó£ìx

[Y,U] = orzPCA(X(:,:),cRate);
uL = unique(L);
nClass = size(uL,2);
nDim = size(Y,1);
Sw = zeros(nDim,nDim);
Sb = zeros(nDim,nDim);
M = mean(Y(:,:),2);
M1 = zeros(nDim,nClass);
for I=1:size(uL,2)    
    S = Y(:,uL(I) == L);    
    Sw = Sw + cov(S',1);    
    M1(:,I) = mean(S,2);
    Sb = Sb + size(S,2)*(M-M1(:,I))*(M-M1(:,I))';
end

nDisDim = min(nClass-1,nDim);
A = Sw\Sb;
[V C] = eigs(A,nDisDim);
[C ind] = sort(diag(C),'descend');
V=V(:,ind);
J = trace((V'*Sw*V)\(V'*Sb*V));

T = V'*U';
Z = orzTransform(T,X);
N = orzTransform(V',M1);


