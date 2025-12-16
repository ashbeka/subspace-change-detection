clear;
load('./CVLABFace2.mat');


% Learning Data: X1
[nDim nNum1 nClass] = size(X1);
%disp([nDim nNum1 nClass])

% Input Data: X2
[nDim nNum2 nSet nClass] = size(X2);
%disp([nDim nNum2 nSet nClass])

% Dimension of reference subspace
nSubDim1 = 20;

% Dimension of input subspace
nSubDim2 = 3;

% Parameter of Gaussian-Kernel
nSigma2 = 0.1;


disp('train');
kmsm = CvtKMSM();
kmsm = kmsm.train(X1, nSubDim1,nSigma2);


disp('test');
SIM = zeros(nClass,nSet,nClass);
for H=1:nSet
    SIM(:,H,:) = kmsm.predict(X2(:,:,H,:), nSubDim2,nSigma2);    
end
SIM = SIM(:,:);
imagesc(SIM);


[var, ind] = max(SIM,[],1);
label = cvtLabel(nSet,nClass);
disp('---RECOGNITION  RATE---')
disp('Kernel Mutual Subspace Method')
disp(strcat('-------',num2str(mean(ind == label) * 100),'% --------'))

