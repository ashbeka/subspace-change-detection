function [ MAP ]= cvtSepEdgeFilter(X,h,w)
% function [ MAP ]= cvtSepEdgeFilter(X,h,w)
% w = 2 h = 3
% MAP(:,:,1)
%        w=2  
%        + + + + +   1,1
%    h=3 + + + + + 
%        + + + + +              3,5
%            X           4,3
%        * * * * *
%        * * * * *
%        * * * * *
%                       ay=y-h, ax=x-w   by=y-1, bx=x+w
% MAP(:,:,2)
%         h=3
%   w=2       3,4
%   1,1  + + +   * * * 
%        + + +   * * *
%        + + + X * * *
%        + + +   * * *
%        + + +   * * *
%           5,3           
%                       ay=y-w, ax=x-h   by=y+w, bx=x-1


%%

[H W]= size(X);

%% 積分イメージ
I = zeros(size(X)+1);
I(2:size(I,1),2:size(I,2)) =  cumsum(cumsum(X,1),2);
%% 二乗積分イメージ
P = zeros(size(X)+1);
P(2:size(P,1),2:size(P,2)) = cumsum(cumsum(X.*X,1),2);

MAP   = zeros([size(X),2]);

r = max([h,(2*w+1)]);
N  = h*(2*w+1);

% ay= y-h, ax=x-w   by=y-1, bx= x+w
for y=1+r:H-r;
   for x=1+r:W-r;     
      S1 =I(y-h,x-w) + I(y-1+1,x+w+1) - I(y-h,x+w+1) -I(y-1+1,x-w);      
      T1 =P(y-h,x-w) + P(y-1+1,x+w+1) - P(y-h,x+w+1) -P(y-1+1,x-w);      
      
      S2 =I(y+1,x-w) + I(y+h+1,x+w+1) - I(y+1,x+w+1) -I(y+h+1,x-w);      
      T2 =P(y+1,x-w) + P(y+h+1,x+w+1) - P(y+1,x+w+1) -P(y+h+1,x-w);      
 
      M = (S1 + S2)/(N+N);
      Y = (T1 + T2)/(N+N);
      St = Y - M^2;

      M1=S1/N;
      M2=S2/N;

      Sb = ((N*((M1-M)^2)) + (N*((M2-M)^2)))/(N+N);
      MAP(y,x,1) = (Sb/St)*sign(M2-M1);

      S1 =I(y-w,x-h) + I(y+w+1,x-1+1) - I(y-w,x-1+1) -I(y+w+1,x-h);      
      T1 =P(y-w,x-h) + P(y+w+1,x-1+1) - P(y-w,x-1+1) -P(y+w+1,x-h);      
      
      S2 =I(y-w,x+1) + I(y+w+1,x+h+1) - I(y-w,x+h+1) -I(y+w+1,x+1);      
      T2 =P(y-w,x+1) + P(y+w+1,x+h+1) - P(y-w,x+h+1) -P(y+w+1,x+1);      
 
      M = (S1 + S2)/(N+N);
      Y = (T1 + T2)/(N+N);
      St = Y - M^2;

      M1=S1/N;
      M2=S2/N;

      Sb = ((N*((M1-M)^2)) + (N*((M2-M)^2)))/(N+N);
      MAP(y,x,2) = (Sb/St)*sign(M2-M1);

   end
end
MAP(isnan(MAP)) = 0;
 