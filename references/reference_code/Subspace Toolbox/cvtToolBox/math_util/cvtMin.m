function [ val INDEX ]= cvtMin(M) 

nSize = size(M);
D = ndims(M);
INDEX  =zeros(1,D);
[val ind ] = min(M(:));
for I=D-1:-1:1
    A = prod(nSize(1:I));
    INDEX(I+1) = ceil(ind/A);
    ind = mod(ind,A);
    if  ind ==0
        ind = A;
    end
end

if ind == 0
    INDEX(1)=A;
else
    INDEX(1) = ind;
end
