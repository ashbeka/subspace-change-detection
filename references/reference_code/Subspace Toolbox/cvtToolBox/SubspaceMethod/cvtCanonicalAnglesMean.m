function C = cvtCanonicalAnglesMean(X,varargin)
% C = cvtCanonicalAnglesMean(X,varargin)
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
    C = zeros(size(X,3),size(X,3));
    for I=1:size(X,3)
        for J=I:size(X,3)
            T = X(:,:,I)'*X(:,:,J);
            C(I,J) = mean(diag(T'*T));
            C(J,I) = C(I,J,:);
        end
    end
end
if nargin == 2
    Y = varargin{1};
    X = X(:,:,:);
    Y = Y(:,:,:);
    C = zeros(size(X,3),size(Y,3));
    for I=1:size(X,3)
        for J=1:size(Y,3)
            T = X(:,:,I)'*Y(:,:,J);
            C(I,J) = mean(diag(T'*T));
        end
    end
end

if nargin > 2
    error('error');
end



