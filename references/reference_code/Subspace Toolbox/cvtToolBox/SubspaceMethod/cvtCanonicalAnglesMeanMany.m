function C = cvtCanonicalAnglesMeanMany(X, Y)
% C = cvtCanonicalAnglesMean(X, Y)
%  äeïîï™ãÛä‘ìØémÇÃê≥èÄäpÇãÅÇﬂÇÈ
%
% ----INPUT----
% X:SubSpace Sets
% Y:SubSpace Sets
%
% ----OUTPUT----
% C:ê≥èÄäp
%
if nargin ~= 2
    error('error');
end

X = X(:,:,:);
Y = Y(:,:,:);

[nDim nDimSubX nSubX] = size(X);
[nDim nDimSubY nSubY] = size(Y);

C = zeros(nSubX, nSubY, nDimSubX, nDimSubY);

for iDimSubX = 1:nDimSubX
for iDimSubY = 1:nDimSubY
    for iSubX = 1:nSubX
    for iSubY = 1:nSubY
        T = X(:,1:iDimSubX,iSubX)' * Y(:,1:iDimSubY,iSubY);
        C(iSubX, iSubY, iDimSubX, iDimSubY) = mean(diag(T'*T));
    end
    end
end
end

% XX = zeros(nDim,sum(1:nDimSubX)*nSubX);
% YY = zeros(nDim,sum(1:nDimSubY)*nSubY);
% 
% offset = 0;
% for iDimSubX = 1:nDimSubX
%     XX(:,offset+(1:(iDimSubX*nSubX))) = cvtMat(X(:,1:iDimSubX,:));
%     offset = offset + iDimSubX*nSubX;
% end
% offset = 0;
% for iDimSubY = 1:nDimSubY
%     YY(:,offset+(1:(iDimSubY*nSubY))) = cvtMat(Y(:,1:iDimSubY,:));
%     offset = offset + iDimSubY*nSubY;
% end
% 
% T = XX' * YY;
% 
% 
% C = zeros(nSubX, nSubY, nDimSubX, nDimSubY);
% 
% offset_y = 0;
% for iDimSubX = 1:nDimSubX
%     offset_x = 0;
%     for iDimSubY = 1:nDimSubY
%         for iSubX = 1:nSubX
%         for iSubY = 1:nSubY
%             Ttmp = T(offset_x+(iSubX-1)*iDimSubX+(1:iDimSubX), offset_y+(iSubY-1)*iDimSubX+(1:iDimSubY));
%             C(iSubX, iSubY, iDimSubX, iDimSubY) = mean(diag(Ttmp'*Tmp));
%         end
%         end
%         offset_x = offset_x + iDumSubX*nSubX;
%     end
%     offset_y = offset_y + iDumSubY*nSubY;
% end
% 
% 
