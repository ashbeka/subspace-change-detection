function [Z ZZZ] = orzGLAC(X,orBin,r)

nDim = 4*(orBin^2);
G = zeros([size(X),2]);

% Roberts Filter
H1 = [1,0;0,-1];
H2 = fliplr(H1);
G(:,:,1) = imfilter(X,H1,'replicate');
G(:,:,2) = imfilter(X,H2,'replicate');

% gradient
%[G(:,:,1) G(:,:,2)] = gradient(X);

% N: gradient magunititude
N = sum(G.^2,3);
% A: gradiennt orientation
A=(atan2(G(:,:,2),G(:,:,1))+pi)*(orBin/(2*pi));
% F: go vector
F = zeros([size(X),orBin]);
for I=1:orBin
    B=1-abs(A-(I-1));
    B(B<0)=0;
    F(:,:,I)=B;
end
F(:,:,1) = F(:,:,1) + abs(sum(F,3)-1);
F = permute(F,[3,1,2]);


ZZ = zeros([size(X),nDim]);
for I=1+r:size(X,1)-r;
    for J=1+r:size(X,2)-r;
        tmpZ = zeros(orBin,4,orBin,1);
        for K=1:orBin
            if F(K,I,J)~=0
                tmpZ(:,1,K) = min(N(I,J),N(I,J+r))   * F(K,I,J)*F(:,I,  J+r);
                tmpZ(:,2,K) = min(N(I,J),N(I+r,J+r)) * F(K,I,J)*F(:,I+r,J+r);
                tmpZ(:,3,K) = min(N(I,J),N(I+r,J  )) * F(K,I,J)*F(:,I+r,J  );
                tmpZ(:,4,K) = min(N(I,J),N(I+r,J-r)) * F(K,I,J)*F(:,I+r,J-r);
            end
        end
        ZZ(I,J,:) = tmpZ(:);
    end
end
ZZZ = ZZ(r+1:end-r,r+1:end-r,:);
Z=sum(sum(ZZZ,1),2);
Z=Z(:);
