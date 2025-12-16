function C = orzCanonicalAngles(X,varargin)

if nargin < 1
    error('error');
end

if nargin == 1
    X = X(:,:,:);
    Y = X(:,:,:);
    
    [nDim nSubDim1,nSet1]  = size(X);
    [nDim nSubDim2,nSet2]  = size(Y);
    B = reshape((X(:,:)'*Y(:,:)).^2,nSubDim1,nSet1,nSubDim2,nSet2);
    
    C =( cumsum(cumsum(B,1),3));
    
    for S1 = 1:nSubDim1
        for S2 = 1:nSubDim2
            C(S1,:,S2,:) = C(S1,:,S2,:)/min([S1,S2]);
        end
    end
   
    C = permute(C,[2,4,1,3]);
 
end
if nargin == 2
    Y = varargin{1};
    
    X = X(:,:,:);
    Y = Y(:,:,:);
    
    [nDim nSubDim1,nSet1]  = size(X);
    [nDim nSubDim2,nSet2]  = size(Y);
    B = reshape((X(:,:)'*Y(:,:)).^2,nSubDim1,nSet1,nSubDim2,nSet2);
    
    C = squeeze( cumsum(cumsum(B,1),3));
    for S1 = 1:nSubDim1
        for S2 = 1:nSubDim2
            C(S1,:,S2,:) = C(S1,:,S2,:)/min([S1,S2]);
        end
    end
    
    C = permute(C,[2,4,1,3]);

end

if nargin > 2
    error('error');
end



