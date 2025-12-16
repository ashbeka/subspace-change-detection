function C = orzCanonicalAngles2(X,Y)


X = X(:,:,:);
Y = Y(:,:,:);

[nDim nSubDim1,nSet1]  = size(X);
[nDim nSubDim2,nSet2]  = size(Y);
B = reshape((X(:,:)'*Y(:,:)),nSubDim1,nSet1,nSubDim2,nSet2);
B = permute(B,[1,3,2,4]);

nSubDim = min([nSubDim1,nSubDim2]);
C = zeros(nSet1,nSet2,nSubDim);
for I1=1:nSet1
    for I2=1:nSet2
        C(I1,I2,:) = cumsum(svd(B(:,:,I1,I2)).^2,1);
    end
end
for I=1:nSubDim
    C(:,:,I) = C(:,:,I)/I;
end
