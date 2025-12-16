function T = TEfeatures2D(V,r,n,catFlag)
[x,y,f] = size(V);

T = zeros(x,y,r,n);

idx = pickrandom(r,f,n);

for i = 1:n
    T(:,:,:,i) = V(:,:,idx(:,i));
end

if catFlag == 1 % concatenate
   T = orzReshape(T,1);
end
end

function idx = pickrandom(r,f,n)
% repeat the following operation n times:
%   pick randomly r numbers from a total of f, and puts them in increasing order.

idx = zeros(r,n);
for i = 1:n
    tp = randperm(f);
    tp = tp(1:r);
    idx(:,i) = sort(tp);
end
end
% function q = vec(q)
% q = q(:);
% end
% pickrandom(r,n)