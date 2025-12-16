function X = cvtRandFloat(nSize,nMin,nMax)
if nargin ~= 3
    error('function X = cvtRandFloat(nSize,nMin,nMax)')
end


if nMax < nMin
   tmp = nMax;
   nMax = nMin;
   nMin = tmp;
end

X= (nMax-nMin +1)*(rand(nSize)/(1+eps)) + nMin;
