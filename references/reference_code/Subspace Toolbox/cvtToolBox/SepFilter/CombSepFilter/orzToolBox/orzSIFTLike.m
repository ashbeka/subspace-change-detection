function S = orzSIFTLike(X,orBin)
%function Z = orzSIFTLike(X,orBin)
% X      : image
% orBin  : orientaion-Bin size
%X = rgb2gray(imread('C:\Users\ohkawa\Desktop\a.png'));

% X= rand(32,32)
% orBin  =16;
% nHs = 8;
% nWs = 8;
% nSlideH =8;
% nSlideW = 8;

F = double(X);
[nH,nW] = size(F);

%勾配計算
[FX FY] = gradient(F);
%勾配強度
M = sqrt(FX.^2+FY.^2);
%勾配方向
A=(atan2(FY,FX)+pi)*(orBin/(2*pi));
A=round(A);
A(A==0)=orBin;
S = zeros([size(F),orBin]);
for B=1:orBin
    S(:,:,B) = M.*(A == B);
end

