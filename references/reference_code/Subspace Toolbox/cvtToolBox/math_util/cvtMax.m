function [ val INDEX ]= cvtMax(M) 

nSize = size(M);
D = ndims(M);
INDEX  =zeros(1,D);
[val ind ] = max(M(:));
for I=D-1:-1:1
    A = prod(nSize(1:I));
    INDEX(I+1) = ceil(ind/A);
    ind = mod(ind,A);
end
INDEX(1) = ind;

