function [f ff] = cvtHlac25(img, r)
B = logical(img);
[h w] = size(B);
dim = 25;

B1 = B(1+r  :h-r  ,1+r  :w-r);   B1=B1(:);%1
B2 = B(1+r  :h-r  ,1+r+r:w-r+r); B2=B2(:);%2
B3 = B(1+r-r:h-r-r,1+r+r:w-r+r); B3=B3(:);%3
B4 = B(1+r-r:h-r-r,1+r  :w-r);   B4=B4(:);%4
B5 = B(1+r-r:h-r-r,1+r-r:w-r-r); B5=B5(:);%5
B6 = B(1+r  :h-r  ,1+r-r:w-r-r); B6=B6(:);%6
B7 = B(1+r+r:h-r+r,1+r-r:w-r-r); B7=B7(:);%7
B8 = B(1+r+r:h-r+r,1+r  :w-r  ); B8=B8(:);%8
B9 = B(1+r+r:h-r+r,1+r+r:w-r+r); B9=B9(:);%9

ff = zeros((h-2*r)*(w-2*r),dim);
ff(:,1) = B1;
ff(:,2) = B1&B2;
ff(:,3) = B1&B3;
ff(:,4) = B1&B4;
ff(:,5) = B1&B5;
ff(:,6)  = B1&B2&B6;
ff(:,7)  = B1&B3&B7;
ff(:,8)  = B1&B4&B8;
ff(:,9)  = B1&B5&B9;
ff(:,10) = B1&B3&B6;
ff(:,11) = B1&B4&B7;
ff(:,12) = B1&B5&B8;
ff(:,13) = B1&B6&B9;
ff(:,14) = B1&B1&B7;
ff(:,15) = B1&B2&B8;
ff(:,16) = B1&B3&B9;
ff(:,17) = B1&B2&B5;
ff(:,18) = B1&B4&B6;
ff(:,19) = B1&B5&B7;
ff(:,20) = B1&B6&B8;
ff(:,21) = B1&B7&B9;
ff(:,22) = B1&B2&B8;
ff(:,23) = B1&B3&B9;
ff(:,24) = B1&B2&B3;
ff(:,25) = B1&B3&B5;

f = sum(ff, 1)';

if nargout >= 2
    ff = reshape(ff',dim,h-2*r,w-2*r);
end

end
