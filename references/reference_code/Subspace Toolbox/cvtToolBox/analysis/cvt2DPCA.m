function[Y, V, D] = cvt2DPCA(X, dim)

sizeX = size(X);
Ctmp=cov(reshape(X, sizeX(1)*sizeX(2), sizeX(3))',1);
C = zeros(sizeX(1),sizeX(1));

for i = sizeX(2)
    index = (1:sizeX(1)) + ((i-1)*sizeX(1));
    C = C + Ctmp(index, index);
end
C = C/sizeX(2);

[V D] = svds(C, dim, 'L');

Y = zeros(dim, sizeX(2), sizeX(3));
for i = 1:sizeX(3)
    Y(:,:,i) = V' * X(:,:,i);
end
