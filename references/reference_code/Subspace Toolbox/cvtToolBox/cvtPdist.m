function Y = cvtPdist(X)
X=X';
D = cvtL2Norm(X,X).^0.5;
nNum = size(X,2);
Y = zeros(1,(nNum-1)*nNum/2);
cnt = 1;
for I1 =1:size(D,1)
   for I2 =I1+1:size(D,2)
      Y(cnt) = D(I1,I2);
      cnt = cnt+1;
   end
end
