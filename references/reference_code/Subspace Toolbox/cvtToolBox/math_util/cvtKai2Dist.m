function d = cvtKai2Dist(X, Y)

numX = size(X,2);
numY = size(Y,2);


d = zeros(numX, numY);
for xi = 1:numX
for yi = 1:numY
    m = 0.5*(X(:,xi)+Y(:,yi));
    m(m==0) = 1;
    d(xi, yi) = sum(((X(:,xi)-m).^2)./ m);
end
end
