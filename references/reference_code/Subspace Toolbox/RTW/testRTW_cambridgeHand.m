
clear
% load('C:\Users\Lincon\Documents\MATLAB\datasets\cambridgeHand\CambridgeHand12x16.mat'); %12core
% load('C:\Users\Lincon\Dropbox\matlab_home\datasets\CambridgeHand12x16.mat') % laptop
load('Z:\Lincon\Datasets\CambridgeHands\CambridgeHand12x16.mat')


%------------pre-processing
DATA = permute(DATA,[1 3 2]);
% nClass = 3;
% idx = randperm(9);
% idx = idx(1:nClass)
% DATA = DATA(:,:,1:nClass);
idx = [9 4 7];
DATA = DATA(:,:,idx);

% DATA = cellfun(@(x) orzReshape(x,1),DATA, 'UniformOutput', false);
X1 = DATA(1:10,:,:);
X2 = DATA(11:20,:,:);

X1 = orzReshape(X1,1);
X2 = orzReshape(X2,1);

%--------------------set up network

% layer properties 
PatchSize = 5;
NumFilters = 5;
PCAconvLayer = SetNetLayer('PCA',PatchSize,NumFilters,[],[]);

% network properties
HistBlockSize = 7;
BlkOverLapRatio = 0;
Classifier = 'GDA';
SubDim = 5;
% 1-layer net
Net = SetNet(PCAconvLayer,HistBlockSize,BlkOverLapRatio,Classifier,[SubDim]);

r = 15; % number of random selections for one TE feature
n = 100; % number of generated TE features
ClassSubDim = 50;
nDimPS = 15;
% Accuracy = deep_RTWmachine(X1,X2,r,n,Net);
%------------pre-processing
DATA2 = cellfun(@(x) orzReshape(x,1),DATA, 'UniformOutput', false);
X1 = DATA2(1:10,:,:);
X2 = DATA2(11:20,:,:);

X1 = orzReshape(X1,1);
X2 = orzReshape(X2,1);


Accuracy2 = RTWmachine(X1,X2,r,n,SubDim);
Accuracy3 = GDS_RTWmachine(X1,X2,r,n,SubDim,ClassSubDim,nDimPS);

