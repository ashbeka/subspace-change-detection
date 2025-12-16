function [C ]= orzKernelCanonicalAngles(X1, A1,varargin)
% function = cvtKernelCanonicalAngles(X1, A1,varargin)
% 
% orzKernelCanonicalAngles(X1, A1,nSigma)
% orzKernelCanonicalAngles(X1, A1, X2, A2, nSigma)
% X1: 
% A1: 
% X2: 
% A2: 
% nSigma: 
% 
% C:
% CC:

if nargin == 3
    X1 = X1(:,:,:);
    A1 = A1(:,:,:);
    nSigma = varargin{1};
    
    [nNum1,nSubDim1,nSet1] = size(A1);
    
    B = zeros(nSet1,nSet1,nSubDim1,nSubDim1);
    for I=1:nSet1
        for J=1:nSet1
            K = exp(-orzL2Distance(X1(:,:,I),X1(:,:,J))/nSigma);                    
            B(I,J,:,:) = (A1(:,:,I)'*K*A1(:,:,J)).^2;
        end
    end
    C =( cumsum(cumsum(B,3),4));
    for S1 = 1:nSubDim1
        for S2 = 1:nSubDim1
            C(:,:,S1,S2) = C(:,:,S1,S2)/min([S1,S2]);
        end
    end
    
elseif nargin == 5
    X2 = varargin{1};
    A2 = varargin{2};
    nSigma = varargin{3};
    
    X1 = X1(:,:,:);
    A1 = A1(:,:,:);
    X2 = X2(:,:,:);
    A2 = A2(:,:,:);
    [nNum1,nSubDim1,nSet1] = size(A1);
    [nNum2,nSubDim2,nSet2] = size(A2);
    
    B = zeros(nSet1,nSet2,nSubDim1,nSubDim2);
    for I=1:nSet1
        for J=1:nSet2          
            K = exp(-orzL2Distance(X1(:,:,I),X2(:,:,J))/nSigma);                    
            B(I,J,:,:) = (A1(:,:,I)'*K*A2(:,:,J)).^2;
        end
    end
    C =( cumsum(cumsum(B,3),4));
    for S1 = 1:nSubDim1
        for S2 = 1:nSubDim2
            C(:,:,S1,S2) = C(:,:,S1,S2)/min([S1,S2]);
        end
    end
    
else
    error('error:function C = cvtKernelCanonicalAngles(X1, A1,varargin)');
end

