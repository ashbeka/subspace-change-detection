% 2010/0624/ Yasuhiro Ohkawa
% This is sample code of Mutual Subspace Method(MSM)
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
nSubDim2 = 5;

% Normalizing all column-vectors of X1 and X2
X1 = sbsNormalize(X1);
X2 = sbsNormalize(X2);

%%  Learning Phase
% Orthonormal basis vectors of Reference Subspaces
V1 = zeros(nDim,nSubDim1,nClass);
D1 = zeros(nSubDim1,nClass);
for I=1:nClass
    % D1: eigen values
       [V1(:,:,I) D1(:,I)] = sbsBasisVector(X1(:,:,I),nSubDim1);
end

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


%%
% Calculating the similarities betweeb input subspace and reference subspaces
% Similaritiy is defined using all canonical angles.
SIM = zeros(nClass,nSet,nClass);
for J=1:nClass
   for H=1:nSet
      for I=1:nClass
         % cos( Canonical Angles ) 
         COS_C_ANGLES = svd(V1(:,:,I)'*V2(:,:,H,J));
%          COS_C_ANGLES = svd(V1(:,:,I)*V2(:,:,H,J)*V1(:,:,I));
%          COS_C_ANGLES = sum(diag(V1(:,:,I)'*V2(:,:,H,J)))^2;

            % similarity = mean(cos^2(c_angles))
         SIM(I,H,J) = mean(COS_C_ANGLES.^2);
      end
   end
end
SIM = SIM(:,:);

%%
[val ind] = max(SIM,[],1);
disp('---RECOGNITION  RATE---')
disp('Mutual Subspace Method')
disp(strcat('-------',num2str(mean(ind == Label) * 100),'% --------'))

%%
imgSize = 16;
F = abs(reshape(sbsNormalize(V1(:,:)),imgSize,imgSize,nSubDim1,nClass));
F = permute(F,[1,4,2,3]);
figure(10)
image(reshape(F,[imgSize*nClass,imgSize*nSubDim1])*255)
title 'Orthonomal Basis Vectors of Each Subspace';
axis equal
colormap bone

