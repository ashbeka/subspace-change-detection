function Param = cvtParamSpan(N1,N2,nBin)

if N1 == N2
   disp('N1 == N2');
   Param = [];
else
   nSpan = ((N2-N1)/(nBin-1));
   Param = N1:nSpan:N2;
end
