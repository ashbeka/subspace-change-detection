% 2010/0520/ Yasuhiro Ohkawa
% This is sample code of Orthogonal Mutual Subspace Method(OMSM)
%% Initialize
clear
load('./CVLABFace2.mat');


% Learning Data: X1
[nDim nNum1 nClass] = size(X1);
%disp([nDim nNum1 nClass])

% Input Data: X2
[nDim nNum2 nSet nClass] = size(X2);
%disp([nDim nNum2 nSet nClass])

% Dimension of reference subspace
nSubDim1 =20;

% Dimension of input subspace
nSubDim2 = 3;

% Normalizing all column-vectors of X1 and X2
X1 = sbsNormalize(X1);
X2 = sbsNormalize(X2);

%% Learning Phase
% Orthonormal basis vectors of Reference Subspaces
V1 = zeros(nDim,nSubDim1,nClass);
D1 = zeros(nSubDim1,nClass);
for I=1:nClass
    % D1: eigen values
       [V1(:,:,I) D1(:,I)] = sbsBasisVector(X1(:,:,I),nSubDim1);
end


%% OrthogonalTransform
P = zeros(nDim,nDim);
for I=1:nClass
   P = P + V1(:,:,I)*V1(:,:,I)';
end
[B C] = eig(P);
C = diag(C);
[C index] = sort(C,'descend');
B = B(:,index);C = C(index);
G = sqrt(inv(diag(C(1:rank(P)))))* B(:,1:rank(P))';


%% Recognition Phase

% Orthonormal basis vectors of Input Subspaces
V2 = zeros(nDim,nSubDim2,nSet,nClass);
D2 = zeros(nSubDim2,nSet,nClass);
for J=1:nClass
   for H=1:nSet
       % D2: eigen values
      [V2(:,:,H,J) D2(:,H,J)] = sbsBasisVector(X2(:,:,H,J),nSubDim2);
   end
end


%% Orthogonal Transform
U1 = zeros(size(G,1),nSubDim1,nClass);
for I=1:nClass
   U1(:,:,I) = orth(G*V1(:,:,I));
end
U2 = zeros(size(G,1),nSubDim2,nSet,nClass);
for J=1:nClass
   for H=1:nSet
      U2(:,:,H,J) = orth(G*V2(:,:,H,J));
   end
end


%%
% Calculating the similarities between input subspace and reference subspaces
% Similaritiy is defined using all canonical angles.
SIM = zeros(nClass,nSet,nClass);
for J=1:nClass
   for H=1:nSet
      for I=1:nClass
         % cos( Canonical Angles ) 
         COS_C_ANGLES = svd(U1(:,:,I)'*U2(:,:,H,J));
         % similarity = mean(cos^2(c_angles))
         SIM(I,H,J) = mean(COS_C_ANGLES.^2);
      end
   end
end
SIM = SIM(:,:);

%%
[val ind] = max(SIM,[],1);
disp('---RECOGNITION  RATE---')
disp('Orthogonal Mutual Subspace Method')
disp(strcat('-------',num2str(mean(ind == Label) * 100),'% --------'))

%%
imgSize = 16;
F = abs(reshape(G'*U1(:,:),imgSize,imgSize,nSubDim1,nClass));
F = permute(F,[1,4,2,3]);
figure(10)
image(reshape(F,[imgSize*nClass,imgSize*nSubDim1])*105)
title 'Visualizing orthonomal Basis Vectors of all the class subspaces transformed by G';
axis equal
colormap bone

