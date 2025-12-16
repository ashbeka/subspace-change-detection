function C = cvtCanonicalAngles(X, Y)
% C = cvtCanonicalAngles(X,varargin)
%  Še•”•ª‹óŠÔ“¯m‚Ì‚Ì³€Šp‚ğ‹‚ß‚é
% 
% ----INPUT----
% X:SubSpace Sets
% Y:SubSpace Sets
% 
% ----OUTPUT----
% C:³€Šp
% 

if nargin < 1
    error('error');
end

if nargin == 1
    X = X(:,:,:);
    C = zeros(size(X,3),size(X,3),size(X,2));
    for I=1:size(X,3)
        for J=I:size(X,3)
            C(I,J,:) = svd(X(:,:,I)'*X(:,:,J)).^2;
            C(J,I,:) = C(I,J,:);
        end
    end
end

if nargin == 2
    X = X(:,:,:);
    Y = Y(:,:,:);
    C = zeros(size(X,3),size(Y,3),min([size(X,2),size(Y,2)]));
    for I=1:size(X,3)
        for J=1:size(Y,3)
            C(I,J,:) = svd(X(:,:,I)'*Y(:,:,J)).^2;
        end
    end
end

if nargin > 2
    error('error');
end



