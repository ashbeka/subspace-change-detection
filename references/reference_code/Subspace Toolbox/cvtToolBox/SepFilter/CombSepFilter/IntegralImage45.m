function Z = cvtIntegralImage45__(X)
%X = double(X);
Z = zeros(size(X,1)+2,size(X,2)+1);
Z(2:end-1,2:end) = X;
tmpX = Z;
for J =3:size(Z,2)
   Z(1,J) = Z(1,J) + Z(1+1,J-1)  + tmpX(1,J-1);
   for I =2:size(Z,1)-1
      Z(I,J) = Z(I,J) + Z(I-1,J-1) + Z(I+1,J-1) - Z(I,J-2) + tmpX(I,J-1);
   end
   Z(end,J) = Z(end,J) +Z(end-1,J-1)  + tmpX(end,J-1);
end
Z=Z(2:end,:);
