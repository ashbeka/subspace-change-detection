function Z = IntegralImage(X)

Z = zeros(size(X)+1);
Z(2:size(Z,1),2:size(Z,2)) =  cumsum(cumsum(X,1),2);
