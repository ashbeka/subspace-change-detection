function Y = cvtTriU(X)
Y = X(logical(triu(ones(size(X,1)),1)));