function [ Oy ] = cvtKomsmTransform(y, X ,L,KO,A,P )

Y = cvtKernelMatrix(X,y,P.sigma2);
a=[];
for c = 1:P.n_class
   a = cat(1,a ,(A(c).A)' * Y(L==c,:));
end
Oy = KO'*a;