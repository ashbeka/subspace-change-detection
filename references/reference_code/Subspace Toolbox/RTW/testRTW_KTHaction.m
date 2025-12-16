clear
load('Z:\Lincon\Datasets\KTH_action_dataset\KTH_dataset.mat')
DATA = permute(kth_data,[1 2 4 3]); % becomes [persons conditions repetitions classes]
X1 = DATA(:,:,1:2,:);
X2 = DATA(:,:,3:4,:);

[X1,TrnLabels] = processKTHdata_RTWexperiments(X1);
[X2,TestLabels] = processKTHdata_RTWexperiments(X2);

% DATA = reshape(DATA,[size(DATA,1)*size(DATA,2)*size(DATA,3),size(DATA,4)]);
% 
% % emptyLogic = cellfun(@isempty,DATA);
% % idx = ind2sub(size(emptyLogic),find(emptyLogic));
% 
% Labels = orzLabel(size(DATA,1),size(DATA,2));
% 
% Labels = Labels(~cellfun('isempty',DATA));
% DATA = DATA(~cellfun('isempty',DATA));
% % for i = 1:13
% %     % kth_data(a(1),b(1),c(1),d(1)) = kth_data(a(1),b(1),c(1)+1,d(1));
% %     DATA(a(i),b(i),c(i),d(i)) = DATA(a(i),b(i),c(i)+1,d(i));
% % end
% 
% newsize = [16 16];
% 
% DATA2 = cellfun(@(x) rgbvid2grayvid(x),DATA, 'UniformOutput', false);
% DATA2 = cellfun(@(x) vidResize(x,newsize),DATA2, 'UniformOutput', false);
% DATA2 = cellfun(@(x) orzReshape(x,1),DATA2, 'UniformOutput', false);


% divided repetitions

% idx = randperm(length(DATA2));
% X1 = DATA2(:,:,1:2,:);
% X2 = DATA2(:,:,3:4,:);

%--------------------set up network

% layer properties 
% PatchSize = 5;
% NumFilters = 5;
% PCAconvLayer = SetNetLayer('PCA',PatchSize,NumFilters,[],[]);
% 
% % network properties
% HistBlockSize = 7;
% BlkOverLapRatio = 0;
% Classifier = 'GDA';
SubDim = 10;
% 1-layer net
% Net = SetNet(PCAconvLayer,HistBlockSize,BlkOverLapRatio,Classifier,[SubDim]);

r = 20;
n = 50;
ClassSubDim = 60;
nDimPS = 10;
% Accuracy = deep_RTWmachine(X1,X2,r,n,Net);
%------------pre-processing
% DATA2 = cellfun(@(x) orzReshape(x,1),DATA, 'UniformOutput', false);
% X1 = DATA2(1:10,:,:);
% X2 = DATA2(11:20,:,:);

% X1 = orzReshape(X1,1);
% X2 = orzReshape(X2,1);


% Accuracy2 = RTWmachine(X1,X2,r,n,SubDim);
Accuracy2 = RTWmachine(X1,X2,TrnLabels,TestLabels,r,n,SubDim);
% Accuracy3 = GDS_RTWmachine(X1,X2,r,n,SubDim,ClassSubDim,nDimPS);
Accuracy3 = GDS_RTWmachine(X1,X2,TrnLabels,TestLabels,r,n,SubDim,ClassSubDim,nDimPS);

