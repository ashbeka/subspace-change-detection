function C = cvtKernelCanonicalAngles(X1, A1,varargin)
% C = cvtKernelCanonicalAngles(X1, A1,varargin)


if nargin == 3
    X1 = X1(:,:,:);
    A1 = A1(:,:,:);
    Sigma2 = cell2mat(varargin(1));
    
    C = zeros(size(A1,3),size(A1,3),size(A1,2));
    for I=1:size(A1,3)
        for J=I:size(A1,3)
            K = cvtKernelMatrix(X1(:,:,I),X1(:,:,J),Sigma2);
            C(I,J,:) = svd(A1(:,:,I)'*K*A1(:,:,J)).^2;
            C(J,I,:) = C(I,J,:);
        end
    end

elseif nargin == 5
    X2 = varargin{1};
    A2 = varargin{2};
    Sigma2 = varargin{3};
    X1 = X1(:,:,:);
    X2 = X2(:,:,:);
    A1 = A1(:,:,:);
    A2 = A2(:,:,:);
    
    C = zeros(size(A1,3),size(A2,3),min([size(A1,2),size(A2,2)]));
    for I=1:size(A1,3)
        for J=1:size(A2,3)
            K = cvtKernelMatrix(X1(:,:,I),X2(:,:,J),Sigma2);
            C(I,J,:) = svd(A1(:,:,I)'*K*A2(:,:,J)).^2;
        end
    end
else 
    error('error:function C = cvtKernelCanonicalAngles(X1, A1,varargin)');
end

