clear
load('./dataset.mat');

%% 学習データの変換
X1 = man1_4_Training;
L1 = reshape(repmat(1:size(X1,3),size(X1,2),1),1,size(X1,2)*size(X1,3));
X1 = X1(:,:);

%% テストデータの変換
X2 = man1_4_Testing;
L2 = reshape(repmat(1:size(X2,3),size(X2,2),1),1,size(X2,2)*size(X2,3));
X2 = X2(:,:);

keep3 X1 X2 L1 L2

n_class = 4;
whos

alpha = 1e-7;
sigma2 = 3e6;

[Y1,A] = cvtKDA(X1,L1,sigma2,alpha);
M1 = zeros(size(Y1,1),n_class);
for I=1:n_class;
   M1(:,I) = mean(Y1(:,L1==I),2);
end
[Y2] = cvtKDA_prj(X2,X1,A,sigma2);

SW = cvtL2Norm(M1,Y2);
[val ind] = min(SW);
rate = mean(L2 == ind);
cvtPlot3(Y1,L1,1)
cvtPlot3(Y2,L2,2)
cvtPlot3([Y1,Y2],[L1,(L2+n_class)],3);
save ansData
disp('識別率：')
disp(rate);