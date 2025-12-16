%% make test dataset
clear;

dir = 'D:\Image\Ojbect\caltech\caltech101\101_ObjectCategories';
strs = {'airplanes', 'beaver', 'ibis', '\windsor_chair'};
nTrain = 20;
nTest = 10;
nClass = 4;


%%
fff = @(img) [cvtCILAC(img,1); cvtCILAC(img,2); cvtCILAC(img,4)];
dim = 900*3;
% fff = @(img) [cvtCILAC(img,1)];
% dim = 900;

train_x = zeros(dim, nTrain, nClass);
train_l = cvtLabel(nTrain, nClass);

for iClass = 1:nClass
for iTrain = 1:nTrain
    filename = fullfile(dir, strs{iClass}, sprintf('image_%04d.jpg', iTrain));
    train_x(:,iTrain,iClass) = fff(imread(filename));
end
end

test_x = zeros(dim, nTest, nClass);
test_l = cvtLabel(nTest, nClass);

for iClass = 1:nClass
for iTest = 1:nTest
    filename = fullfile(dir, strs{iClass}, sprintf('image_%04d.jpg', iTest+nTrain));
    test_x(:,iTest,iClass) = fff(imread(filename));
end
end

train_x = train_x(:,:);
test_x = test_x(:,:);

%%
%score = cvtTestMDA(train_x, train_l, test_x, test_l, 0.9999);

[train_y pca_v] = cvtPCA(train_x, 0.999);
[train_y mda_v] = cvtMDA(train_y, train_l, 0.999);
class_m = cvtClassMean(train_y, train_l);
% ddd = cvtL2Norm(mda_v'*pca_v'*test_x, train_y);
% [~,predict_index] = min(ddd,[],2);
% predict_l = train_l(predict_index);
% disp(mean(predict_l(:) == test_l(:)));

ddd = cvtL2Norm(train_y, class_m);
[~,predict_l] = min(ddd,[],2);
disp(mean(predict_l(:) == train_l(:)));



