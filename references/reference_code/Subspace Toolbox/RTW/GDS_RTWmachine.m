function Accuracy = GDS_RTWmachine(X1,X2,k,n,nSubDim,ClassSubDim,nDimPS)
% function Accuracy = GDS_RTWmachine(X1,X2,TrnLabels,TestLabels,k,n,nSubDim,ClassSubDim,nDimPS)

% choose k frames and repeat the process n times

% [nDim,nSampPerSet,nSet,nClass] = size(X2);

T1 = DatasetTEfeatures(X1,k,n);
T2 = DatasetTEfeatures(X2,k,n);

[Accuracy,Time,S] = GDS_GrasMachine2(T1,T2,nSubDim,ClassSubDim,nDimPS);
% [Accuracy,Time,S] = GDS_GrasMachineUnbalanced(T1,T2,TrnLabels,TestLabels,nSubDim,ClassSubDim,nDimPS);


% [V1, D1] = DatasetBasisVector(T1,nSubDim);
% [V2, D2] = DatasetBasisVector(T2,nSubDim);
% 
% [TrnSub4GDS,~] = DatasetBasisVector(V1,ClassSubDim);
% G = GetGDSoperator(TrnSub4GDS,nDimPS);
% V1 = Projecton2GDS(G,V1);
% SIMTrainTrain = GetSim(V1);
% % Time.trn = toc;
% 
% % tic
% V2 = orzReshape(V2,3);
% V2 = Projecton2GDS(G,V2);
% SIMTrainTest = GetSim(V1,V2);
% % Time.test = toc;
% 
% TrnLabels = orzLabel(nTrnSet,nClass);
% TestLabels = orzLabel(nSet,nClass);
% 
% 
% [Accuracy,S] = computeGDANewKNN(SIMTrainTrain, SIMTrainTest, TrnLabels, TestLabels);
% [Accuracy,Time,S] = GrasMachineNewKNN(T1,T2,nSubDim);
end

function T = DatasetTEfeatures(X,r,n)
if size(X,2) > 1
    szX = size(X);
    X = X(:);
else
    szX = length(X);
end
nVids = length(X);
T = cell(nVids,1);
for i = 1:nVids
    T{i} = TEfeatures(X{i},r,n,1); % concatenate
end
T = cat(3,T{:});
szT = size(T);
T = reshape(T,[szT(1:2),szX]);
end