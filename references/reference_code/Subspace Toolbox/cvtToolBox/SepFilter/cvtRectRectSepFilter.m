function [MAP1 MAP2] = cvtRectRectSepFilter(X,bh,bw,sh,sw,dh,dw)
% function [MAP1 MAP2] = cvtRectRectSepFilter(X,bh,bw,sh,sw,dh,dw)
%
% 引数
% X:         データ，マップ
% bh,bw：    大きな矩形のサイズ
% sh,sw：    小さな矩形のサイズ
% dh,dw：    小さな矩形のずれ
%
% 戻り値
% MAP1:      符号付分離度MAP
% MAP2:      差分MAP

%% 

A  = ones(1+2*bh,1+2*bw);
B  = ones(1+2*bh,1+2*bw);
A(bh+1-sh+dh:bh+1+sh+dh,bw+1-sw+dw:bw+1+sw+dw) = 0;
if size(A) ~= size(B)  
   error('error')
end

%%

[H W]= size(X);

%% 積分イメージ
I = zeros(size(X)+1);
I(2:size(I,1),2:size(I,2)) =  cumsum(cumsum(X,1),2);
%% 二乗積分イメージ
P = zeros(size(X)+1);
P(2:size(P,1),2:size(P,2)) = cumsum(cumsum(X.*X,1),2);

MAP1   = zeros(size(X));
MAP2   = zeros(size(X));

r = max([bh,bw]);
N  = (2*bh+1)*(2*bw+1);
N1 = (2*sh+1)*(2*sw+1);
N2 = N-N1;


for y=1+r:H-r;
   for x=1+r:W-r;     
      S =I(y-bh,x-bw) + I(y+bh+1,x+bw+1) - I(y-bh,x+bw+1) -I(y+bh+1,x-bw);
      T =P(y-bh,x-bw) + P(y+bh+1,x+bw+1) - P(y-bh,x+bw+1) -P(y+bh+1,x-bw);

      M = S/N;
      Y = T/N;
      St = Y - M^2;
      S1 =I(y-sh+dh,x-sw+dw) + I(y+sh+dh+1,x+sw+dw+1) - I(y-sh+dh,x+sw+dw+1) - I(y+sh+dh+1,x-sw+dw);
      S2=S-S1;
      M1=S1/N1;
      M2=S2/N2;

      Sb = ((N1*((M1-M)^2)) + (N2*((M2-M)^2)))/N;
      MAP1(y,x) = (Sb/St)*sign(M2-M1);
      MAP2(y,x)= M2-M1;
   end
end
MAP1(isnan(MAP1))=0;
MAP1(isinf(MAP1))=0;
