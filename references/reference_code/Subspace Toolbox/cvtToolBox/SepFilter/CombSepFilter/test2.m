clear
cd('C:\MATLAB\Ohkawasan\CombSepFilter\CombSepFilter\')
addpath('C:\MATLAB\Ohkawasan\CombSepFilter\CombSepFilter\orzToolBox')


load('./EyeNonEye.mat')
size(X)
nDim = 9*8*8;
Z = zeros(nDim,size(X,3));
for I=1:size(X,3)
    Y = X(:,:,I);
    Z(:,I) = HOG(Y/norm(Y(:)));
end
[Y V D M] = orzMDA(Z,L,0.999);

EYE.M = M;
EYE.V = V;

keep3 EYE n*;


load('Z:\cvtMATLAB\MVA2011\face240\face240_100x3.mat') %240 x 240 x 100 x 3 person 
DATA=double(X(:,:,:))/255;
nR = 7;
nTH = 0.3;
A=-3*nR:3*nR;
B=-nR:nR;
nS = 240;
J=87
for J=1:100
    %X=DATA(:,:,J);
    img=imread(['C:\MATLAB\2011\dataset\mygaze\eyegazedataset2\res0\face\img-5-01-',sprintf('%.3d.png',J)]);
    X= double( imresize(img,240/size(img,1)))/255;
X=X(1:239,1:239);
clear P
    tic
    I1 = IntegralImage(X);
    P1 = IntegralImage(X.^2);
    I2 = IntegralImage45(X);
    P2 = IntegralImage45(X.^2);
    T(1) = toc;
    tic
    P(:,:,1:2) = CombSimpRectFilter(I1,P1,nR);
    P(:,:,3:4) = CombSimpRectFilter45(I2,P2,nR);
    P(P<=0)=0;
    M = mean(P,3);
    T(2) = toc;
    tic
    PL = cvtFindLocalPeakX(M,1,nTH);
    PL=PL(:,PL(1,:) > nR*3);
    PL=PL(:,PL(2,:) > nR*3);
    PL=PL(:,PL(1,:) < nS-nR*3);
    PL=PL(:,PL(2,:) < nS-nR*3);


    S = cat(3,X,X,X);
    for H=1:size(PL,2)
        S(B+PL(1,H),  PL(2,H),:)=1;
        S(  PL(1,H),B+PL(2,H),:)=1;
    end
    T(3) = toc;
    tic
    Z = zeros(nDim,size(PL,2));
    for H=1:size(PL,2)
        Y=X(A+PL(1,H),A+PL(2,H));
        Z(:,H) = HOG(Y/norm(Y(:)));
    end
    T(4) = toc;
    tic
    D=orzL2Distance(EYE.M,EYE.V*Z);
    [val IDX] = min(D,[],2);

    S(B+PL(1,IDX(1)),  PL(2,IDX(1)),:)=0;
    S(  PL(1,IDX(1)),B+PL(2,IDX(1)),:)=0;
    S(B+PL(1,IDX(1)),  PL(2,IDX(1)),1)=1;
    S(  PL(1,IDX(1)),B+PL(2,IDX(1)),1)=1;

    S(B+PL(1,IDX(2)),  PL(2,IDX(2)),:)=0;
    S(  PL(1,IDX(2)),B+PL(2,IDX(2)),:)=0;
    S(B+PL(1,IDX(2)),  PL(2,IDX(2)),2)=1;
    S(  PL(1,IDX(2)),B+PL(2,IDX(2)),2)=1;
    T(5) = toc;
    
    imshow(S)
    pause(0.01)
end
