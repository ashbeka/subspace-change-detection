% 2010/0624/ Yasuhiro Ohkawa
% This is sample code of Kernel Mutual Subspace Method(KMSM)
clear
load('./CVLABFace2.mat');

% Learning  Data: X1
[nDim nNum1 nClass] = size(X1);

% Input  Data: X2
[nDim nNum2 nSet nClass] = size(X2);

% Dimension of reference subspace
nSubDim1 = 20;

% Dimension of input subspace
nSubDim2 = 3;

% Parameter of Gaussian-Kernel
nSigma2 = 0.1;

% Normalized L2-norm all column-vectors
X1 = sbsNormalize(X1);
X2 = sbsNormalize(X2);

%% Learning Phase

% Combination-Coefficient of Orthonomal basis vectors of Reference Subspaces
A1 = zeros(nNum1,nSubDim1,nClass);
D1 = zeros(nSubDim1,nClass);
for I=1:nClass
    % D1 :eigen value
    [A1(:,:,I) D1(:,I)] = sbsKernelBasisVector(X1(:,:,I),nSubDim1,nSigma2);
end

%% Recognition Phase
% Combination-Coefficient of Orthonomal basis vectors of Input Subspaces
A2 = zeros(nNum2,nSubDim2,nSet,nClass);
D2 = zeros(nSubDim2,nClass);
for J=1:nClass
    for H=1:nSet
        % D2 :eigen value
        [ A2(:,:,H,J) D2(:,H,J)] = sbsKernelBasisVector(X2(:,:,H,J),nSubDim2,nSigma2);
    end
end

% Calculating the similarities betweeb input subspace and reference subspaces
% Similaritiy is defined using all canonical angles.
SIM = zeros(nClass,nSet,nClass);
for J=1:nClass
    for H=1:nSet
        for I=1:nClass
            % cos( Canonical Angles )
            K = sbsKernelMatrix(X1(:,:,I),X2(:,:,H,J),nSigma2);
            COS_C_ANGLES = svd(A1(:,:,I)'*K*A2(:,:,H,J));
            % similarity = mean(cos^2(c_angles))
            SIM(I,H,J) = mean(COS_C_ANGLES.^2);
        end
    end
end
SIM = SIM(:,:);
%%
[val ind] = max(SIM,[],1);
disp('---RECOGNITION  RATE---')
disp('Kernel Mutual Subspace Method')
disp(strcat('-------',num2str(mean(ind == Label) * 100),'% --------'))
