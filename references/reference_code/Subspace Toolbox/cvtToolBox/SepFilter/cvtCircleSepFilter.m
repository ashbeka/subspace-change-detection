function [MAP MASK] = cvtCircleSepFilter(img, r, wi, wo)
%MAP = cvtCircleSepFilter(img, r, wi, wo)
% •ª—£“xƒ}ƒbƒv‚ðì¬
% input
%  img: matrix of (y~x), ‰æ‘œƒf[ƒ^
%  r: scalar, ‰~‚Ì”¼Œa
%  wi: scalar, “à‰~‚ÌŒú‚Ý
%  wo: scalar, ŠO‰~‚ÌŒú‚Ý
% output
%  MAP: matrix of (y~x), •ª—£“xƒ}ƒbƒv
% It was written by ohkawa on July 7,2009. 

[H W] = size(img);
X = double(img);
MAP = zeros(size(X));

L = 2 * (r + wo) +1;
c = r+wo+1;
N1 = 0;
N2 = 0;
List1=zeros(L^2,2);
List2=zeros(L^2,2);
MASK = zeros(L,L);
for py=1:L;
   for px = 1:L;
      if (r^2) >= ((c-py)^2 + (c-px)^2) && (r-wi)^2 <= ((c-py)^2 + (c-px)^2)
         MASK(py,px) = 0.5;
         N1 = N1 + 1;
         List1(N1,:) =[py,px];
      elseif (r+wo)^2 >= ((c-py)^2 + (c-px)^2) && (r^2) <= ((c-py)^2 + (c-px)^2)
         MASK(py,px) = 1;
         N2 = N2 + 1;
         List2(N2,:) =[py,px];
      end
   end
end

List1 = List1(1:N1,:);
List2 = List2(1:N2,:);

N = N1 + N2;

V1 = zeros(N1,1);
V2 = zeros(N2,1);
for y=1:H-L+1;
   for x=1:W-L+1;
      for l = 1:size(List1,1)
         V1(l) = X(List1(l,1)+y-1,List1(l,2)+x-1);
      end
      for l = 1:size(List2,1)
         V2(l) = X(List2(l,1)+y-1,List2(l,2)+x-1);
      end

      M = mean([V1;V2]);
      St = cov([V1;V2],1);
      if St == 0
         MAP(y+c-1,x+c-1) = 0;
      else
         M1 = mean(V1);
         M2 = mean(V2);
         Sb = ((N1*((M1-M)^2)) + (N2*((M2-M)^2)))/N;
         MAP(y+c-1,x+c-1) = Sb/St;
      end
   end
end
