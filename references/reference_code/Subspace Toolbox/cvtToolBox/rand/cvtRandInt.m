function X = cvtRandInt(nSize,nMin,nMax)
if nargin ~= 3
    error('function X = cvtRandInt(nSize,nMin,nMax)')
end

nMax = round(nMax);
nMin = round(nMin);

if nMax < nMin
   tmp = nMax;
   nMax = nMin;
   nMin = tmp;
end

X=floor( (nMax-nMin +1)*(rand(nSize)/(1+eps))) + nMin;