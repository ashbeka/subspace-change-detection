function Accuracy = simplifiedGDS_RTWmachine(TrnSub,TestSub,ClassSubDim,nDimPS)

% [Accuracy,Time,S] = GDS_GrasMachine2(T1,T2,nSubDim,ClassSubDim,nDimPS);


% [TrnSub4GDS,~] = DatasetBasisVector(TrnSub,ClassSubDim);
nTrnSet = size(TrnSub,3);
nSet = size(TestSub,3);
nClass = size(TrnSub,4);

X = orzReshape(TrnSub,3);

sx = size(X);
S = zeros([sx(1),sx(1),sx(3)]);
TrnSub4GDS = zeros([sx(1),ClassSubDim,sx(3)]);
for c = 1:sx(3)
    S(:,:,c) = X (:,:,c)* X(:,:,c)';
    [TrnSub4GDS(:,:,c),d] = eigs(S(:,:,c),ClassSubDim);
end

G = GetGDSoperator(TrnSub4GDS,nDimPS);
TrnSub = orzReshape(TrnSub,3);
TrnSub = Projecton2GDS(G,TrnSub);
SIMTrainTrain = GetSim(TrnSub);
% Time.trn = toc;

% tic
TestSub = orzReshape(TestSub,3);
TestSub = Projecton2GDS(G,TestSub);
SIMTrainTest = GetSim(TrnSub,TestSub);
% Time.test = toc;

TrnLabels = orzLabel(nTrnSet,nClass);
TestLabels = orzLabel(nSet,nClass);


[Accuracy,S] = computeGDANewKNN(SIMTrainTrain, SIMTrainTest, TrnLabels, TestLabels);
% [Accuracy,Time,S] = GrasMachineNewKNN(T1,T2,nSubDim);
end

function T = DatasetTEfeatures(X,r,n)
if size(X,2) > 1
    szX = size(X);
    X = X(:);
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