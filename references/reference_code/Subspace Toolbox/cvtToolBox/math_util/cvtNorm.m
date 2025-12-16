% Euclidean distance matrix between column vectors in X and Y.
%                                   from knn_old.m in somtoolbox
%				    ver. 1.00
% d(i, j) = norm(x(i) - y(j), n)

function d = cvtNorm(X, Y, n)

numX = size(X,2);
numY = size(Y,2);

d = zeros(numX, numY);
for xi = 1:numX
    xx=X(:,xi);
    for yi = 1:numY
        d(xi,yi) = norm(xx - Y(:,yi), n);
    end
end
