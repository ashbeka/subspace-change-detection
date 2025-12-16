function X = orzSubRegion(Z,nSH,nSW)
%function X = orzSubRegion(Z,nSH,nSW)
% Z: [nH nW nDim]ÇÃì¡í•ó âÊëú
% nSHÅFècÇÃï™äÑêî
% nSWÅFâ°ÇÃï™äÑêî
Z=Z(:,:,:);
[nH nW nDim] = size(Z);
nWH = ceil(nH/nSH);
nWW = ceil(nW/nSW);

X = zeros(nDim,nSH,nSW);
for H=1:nSH
    tmpH = 1 + floor((H-1)*(nH/nSH)): nWH + floor((H-1)*(nH/nSH));
    for W=1:nSW
        tmpW = 1 + floor((W-1)*(nW/nSW)): nWW + floor((W-1)*(nW/nSW));
        ZZ = sum(sum(Z(tmpH,tmpW,:),1),2);
        X(:,H,W) = ZZ(:);
    end
end
