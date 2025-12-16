% Euclidean distance matrix between column vectors in X and Y.
%                                   from knn_old.m in somtoolbox
%				    ver. 1.00
% d(i, j) = norm(x(i) - y(j))^2

function d = cvtL2Norm(X, Y)
X = X';
Y = Y';
U = ~isnan(Y); %Y(~U) = 0;
V = ~isnan(X); %X(~V) = 0;
d = abs(X.^2*U'+V*Y'.^2-2*X*Y');
return;
