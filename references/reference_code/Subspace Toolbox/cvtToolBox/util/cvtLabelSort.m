function K = cvtLabelSort(L)

uN = size(unique(L),2);
cL = L;
C = zeros(1,uN);
for I = 1:uN
   C(I) =cL(1) ;
   cL(cL==cL(1) )=[];
end
K = zeros(size(L));
for I = 1:uN
   K(L==C(I)) = I;
end

