function [Y, V, D] = cvtIdentityKernelKLE(X,sdim)

[dim,num] = size(X);
K = size(num,num);
for I=1:num
   for J=I:num
      K(I,J) = dot(X(:,I),X(:,J));
      K(J,I) = K(I,J);
   end
end

sdim = min([num,dim,sdim]);
[A,B] = eigs(K,sdim);
A = A/sqrt(B);
V = X*A;
Y = V'*X;
D = diag(B)/num;