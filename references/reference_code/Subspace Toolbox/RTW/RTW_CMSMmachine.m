% function Accuracy = GDS_RTWmachine(X1,X2,k,n,nSubDim,ClassSubDim,nDimPS)
function Results = RTW_CMSMmachine(X1,X2,TrnLabels,TestLabels,k,n,SubDim,ClassSubDim,nDimPS)

% choose k frames and repeat the process n times

% [nDim,nSampPerSet,nSet,nClass] = size(X2);

T1 = DatasetTEfeatures(X1,k,n);
T2 = DatasetTEfeatures(X2,k,n);

% [Results,Time,S] = GDS_GrasMachineUnbalanced(T1,T2,TrnLabels,TestLabels,nSubDim,ClassSubDim,nDimPS);
[Results,Time,S] = CMSM_MachineUnbalanced(T1,T2,TrnLabels,TestLabels,SubDim,ClassSubDim,nDimPS);
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