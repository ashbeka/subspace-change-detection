function Y = cvtDimFusion(X,No)


L = ndims(X);


if No >= L || No < 1
    Y = X;
else
    A = size(X);
    B = zeros(1,L-1);
    B(1:No-1) = A(1:No-1);
    B(No) = A(No)*A(No+1);
    B(No+1:end) = A(No+2:end);
    Y = reshape(X,B);
end
