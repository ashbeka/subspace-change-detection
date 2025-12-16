function Y = cvtChangeColorMap(X,cMap)

Z = X(:);
A = Z-min(Z);
B = uint8(round(255*(A/max(A))));
Y = reshape(cMap(B+1,:),[size(X),3]);
