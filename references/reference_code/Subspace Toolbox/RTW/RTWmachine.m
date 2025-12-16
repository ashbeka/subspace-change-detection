function [Accuracy,V1,V2] = RTWmachine(X1,X2,r,n,nSubDim)

T1 = DatasetTEfeatures(X1,r,n);
T2 = DatasetTEfeatures(X2,r,n);

% [V1, D1] = DatasetBasisVector(T1,nSubDim);
% [V2, D2] = DatasetBasisVector(T2,nSubDim);

[Accuracy,Time,V1,V2] = GrasMachineNewKNN(T1,T2,nSubDim);

% Accuracy =[];
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