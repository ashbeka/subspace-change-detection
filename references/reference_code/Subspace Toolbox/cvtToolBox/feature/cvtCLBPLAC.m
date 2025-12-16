function [F1 F0] = cvtCLBPLAC(imgs, s, st, r, rt, num_block_h, num_block_w, num_block_t)
%
% dim : F1 = 468 (6*6*13)
%
if ~exist('num_block_h','var'), num_block_h = 1; end
if ~exist('num_block_w','var'), num_block_w = 1; end
if ~exist('num_block_t','var'), num_block_t = 1; end

Z = double(imgs);
[h w t] = size(Z);

C = Z(1+s:h-s,1+s:w-s,1+st:t-st);
X = zeros(6,h-2*s,w-2*s,t-2*st);
X(1,:,:) = Z(1+s+s:h-s+s, 1+s  :w-s  , 1+st   :t-st   ) - C;
X(2,:,:) = Z(1+s-s:h-s-s, 1+s  :w-s  , 1+st   :t-st   ) - C;
X(3,:,:) = Z(1+s  :h-s  , 1+s+s:w-s+s, 1+st   :t-st   ) - C;
X(4,:,:) = Z(1+s  :h-s  , 1+s-s:w-s-s, 1+st   :t-st   ) - C;
X(5,:,:) = Z(1+s  :h-s  , 1+s  :w-s  , 1+st+st:t-st+st) - C;
X(6,:,:) = Z(1+s  :h-s  , 1+s  :w-s  , 1+st-st:t-st-st) - C;
X = double(X>0);

[dd hh ww tt] = size(X);
D  = X(:,1+r  :hh-r  , 1+r  :ww-r  , 1+rt   :tt-rt);
Y1 = X(:,1+r-r:hh-r-r, 1+r-r:ww-r-r, 1+rt-rt:tt-rt-rt);
Y2 = X(:,1+r  :hh-r  , 1+r-r:ww-r-r, 1+rt-rt:tt-rt-rt);
Y3 = X(:,1+r+r:hh-r+r, 1+r-r:ww-r-r, 1+rt-rt:tt-rt-rt);
Y4 = X(:,1+r-r:hh-r-r, 1+r  :ww-r  , 1+rt-rt:tt-rt-rt);
Y5 = X(:,1+r  :hh-r  , 1+r  :ww-r  , 1+rt-rt:tt-rt-rt);
Y6 = X(:,1+r+r:hh-r+r, 1+r  :ww-r  , 1+rt-rt:tt-rt-rt);
Y7 = X(:,1+r-r:hh-r-r, 1+r+r:ww-r+r, 1+rt-rt:tt-rt-rt);
Y8 = X(:,1+r  :hh-r  , 1+r+r:ww-r+r, 1+rt-rt:tt-rt-rt);
Y9 = X(:,1+r+r:hh-r+r, 1+r+r:ww-r+r, 1+rt-rt:tt-rt-rt);
Y10= X(:,1+r-r:hh-r-r, 1+r-r:ww-r-r, 1+rt   :tt-rt);
Y11= X(:,1+r  :hh-r  , 1+r-r:ww-r-r, 1+rt   :tt-rt);
Y12= X(:,1+r+r:hh-r+r, 1+r-r:ww-r-r, 1+rt   :tt-rt);
Y13= X(:,1+r-r:hh-r-r, 1+r  :ww-r  , 1+rt   :tt-rt);

[ddd hhh www ttt] = size(D);
hhh = floor(hhh/num_block_h)*num_block_h;
www = floor(www/num_block_w)*num_block_w;
ttt = floor(ttt/num_block_t)*num_block_t;
size_block_h = hhh / num_block_h;
size_block_w = www / num_block_w;
size_block_t = ttt / num_block_t;

F1 = zeros(ddd, ddd, 6, num_block_h, num_block_w, num_block_t);
F0 = zeros(ddd, num_block_h, num_block_w, num_block_t);

for nti = 1:num_block_t
    tlist = (1:size_block_t)+(nti-1)*size_block_t;
    for nhi = 1:num_block_h
        ylist = (1:size_block_h)+(nhi-1)*size_block_h;
        for nwi = 1:num_block_w
            xlist = (1:size_block_w)+(nwi-1)*size_block_w;
            
            Dtmp  =  D(:, ylist, xlist, tList);
            Y1tmp = Y1(:, ylist, xlist, tList);
            Y2tmp = Y2(:, ylist, xlist, tList);
            Y3tmp = Y3(:, ylist, xlist, tList);
            Y4tmp = Y4(:, ylist, xlist, tList);
            Y5tmp = Y5(:, ylist, xlist, tList);
            Y6tmp = Y6(:, ylist, xlist, tList);
            Y7tmp = Y7(:, ylist, xlist, tList);
            Y8tmp = Y8(:, ylist, xlist, tList);
            Y9tmp = Y9(:, ylist, xlist, tList);
            Y10tmp = Y10(:, ylist, xlist, tList);
            Y11tmp = Y11(:, ylist, xlist, tList);
            Y12tmp = Y12(:, ylist, xlist, tList);
            Y13tmp = Y13(:, ylist, xlist, tList);
            
            Dtmp = Dtmp(:,:);
            F1(:,:,1,nhi,nwi) = Y1tmp(:,:)*Dtmp';
            F1(:,:,2,nhi,nwi) = Y2tmp(:,:)*Dtmp';
            F1(:,:,3,nhi,nwi) = Y3tmp(:,:)*Dtmp';
            F1(:,:,4,nhi,nwi) = Y4tmp(:,:)*Dtmp';
            F1(:,:,5,nhi,nwi) = Y5tmp(:,:)*Dtmp';
            F1(:,:,6,nhi,nwi) = Y6tmp(:,:)*Dtmp';
            F1(:,:,7,nhi,nwi) = Y7tmp(:,:)*Dtmp';
            F1(:,:,8,nhi,nwi) = Y8tmp(:,:)*Dtmp';
            F1(:,:,9,nhi,nwi) = Y9tmp(:,:)*Dtmp';
            F1(:,:,10,nhi,nwi) = Y10tmp(:,:)*Dtmp';
            F1(:,:,11,nhi,nwi) = Y11tmp(:,:)*Dtmp';
            F1(:,:,12,nhi,nwi) = Y12tmp(:,:)*Dtmp';
            F1(:,:,13,nhi,nwi) = Y13tmp(:,:)*Dtmp';
            
            Xtmp = X(:,ylist+r, xlist+r, tlist+rt);
            F0(:,nhi,nwi) = sum(Xtmp(:,:), 2);
        end
    end
end

F0 = F0(:);
F1 = F1(:);

end
