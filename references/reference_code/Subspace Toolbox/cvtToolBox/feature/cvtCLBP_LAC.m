function [ F1 F0 ] = cvtCLBP_LAC(Z, s, r, dt, option)
% function f = cvtCLBP_LAC(imgs, s, r, dt, option)
% dim=3328
% 

if ~exist('option', 'var')
    option = 1;
end
Z = imgs;
[h w t] = size(Z);

if option == 0
    
    
elseif option == 1
    
    C = Z(1+s:h-s, 1+s:w-s, :);
    X = zeros(4, h-2*s, w-2*s, t);
    X(1,:,:,:) = Z(1+s  :h-s  , 1+s+s:w-s+s, :) - C;
    X(2,:,:,:) = Z(1+s-s:h-s-s, 1+s  :w-s  , :) - C;
    X(3,:,:,:) = Z(1+s  :h-s  , 1+s-s:w-s-s, :) - C;
    X(4,:,:,:) = Z(1+s+s:h-s+s, 1+s  :w-s  , :) - C;
    
elseif option == 2

    C = Z(1+s:h-s, 1+s:w-s, :);
    X = zeros(4, h-2*s, w-2*s, t);
    X(1,:,:,:) = Z(1+s-s:h-s-s, 1+s-s:w-s-s, :) - C;
    X(2,:,:,:) = Z(1+s-s:h-s-s, 1+s+s:w-s+s, :) - C;
    X(3,:,:,:) = Z(1+s+s:h-s+s, 1+s-s:w-s-s, :) - C;
    X(4,:,:,:) = Z(1+s+s:h-s+s, 1+s+s:w-s+s, :) - C;
  
end

X=double(X>0);
A=reshape([1,2,4,8]*X(:,:), h-2*s, w-2*s, t)+1;
[hh ww tt] = size(A);
D  =(A(1+r  :hh-r  ,1+r  :ww-r  , 1   :tt-dt) - 1) * 16;
Y1 = A(1+r  :hh-r  ,1+r+r:ww-r+r, 1   :tt-dt) + D;
Y2 = A(1+r+r:hh-r+r,1+r-r:ww-r-r, 1   :tt-dt) + D;
Y3 = A(1+r+r:hh-r+r,1+r  :ww-r  , 1   :tt-dt) + D;
Y4 = A(1+r+r:hh-r+r,1+r+r:ww-r+r, 1   :tt-dt) + D;
Y5 = A(1+r-r:hh-r-r,1+r-r:ww-r-r, 1+dt:tt) + D;
Y6 = A(1+r-r:hh-r-r,1+r  :ww-r  , 1+dt:tt) + D;
Y7 = A(1+r-r:hh-r-r,1+r+r:ww-r+r, 1+dt:tt) + D;
Y8 = A(1+r  :hh-r  ,1+r-r:ww-r-r, 1+dt:tt) + D;
Y9 = A(1+r  :hh-r  ,1+r  :ww-r  , 1+dt:tt) + D;
Y10= A(1+r  :hh-r  ,1+r+r:ww-r+r, 1+dt:tt) + D;
Y11= A(1+r+r:hh-r+r,1+r-r:ww-r-r, 1+dt:tt) + D;
Y12= A(1+r+r:hh-r+r,1+r  :ww-r  , 1+dt:tt) + D;
Y13= A(1+r+r:hh-r+r,1+r+r:ww-r+r, 1+dt:tt) + D;

F = zeros(16*16, 13);
F(:,1) = hist(Y1(:), 1:(16*16));
F(:,2) = hist(Y2(:), 1:(16*16));
F(:,3) = hist(Y3(:), 1:(16*16));
F(:,4) = hist(Y4(:), 1:(16*16));
F(:,5) = hist(Y5(:), 1:(16*16));
F(:,6) = hist(Y6(:), 1:(16*16));
F(:,7) = hist(Y7(:), 1:(16*16));
F(:,8) = hist(Y8(:), 1:(16*16));
F(:,9) = hist(Y9(:), 1:(16*16));
F(:,10)= hist(Y10(:),1:(16*16));
F(:,11)= hist(Y11(:),1:(16*16));
F(:,12)= hist(Y12(:),1:(16*16));
F(:,13)= hist(Y13(:),1:(16*16));
F1 = F(:);
F0 = hist(A, 1:16); 
end