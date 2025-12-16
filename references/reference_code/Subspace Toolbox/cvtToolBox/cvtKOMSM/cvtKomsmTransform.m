function [ Oy ] = cvtKomsmTransform(y, X ,L,KO,A,P )

Y = cvtKernelMatrix(X,y,P.Sigma2);
a=[];
for c = 1:P.nClass
   a = cat(1,a ,(A(c).A)' * Y(L==c,:));
end
Oy = KO'*a;