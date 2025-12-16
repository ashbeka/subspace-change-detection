function [f ff] = cvtHlac35( img,r )
%GRAYHLAC この関数の概要をここに記述
%   詳細説明をここに記述
if ~isa(img, 'double')
    img = im2double(img);
end
dim = 35;

[h w] = size(img);
img1 = img(1+r  :h-r  ,1+r  :w-r  ); img1=img1(:);
img2 = img(1+r  :h-r  ,1+r+r:w-r+r); img2=img2(:);
img3 = img(1+r-r:h-r-r,1+r+r:w-r+r); img3=img3(:);
img4 = img(1+r-r:h-r-r,1+r  :w-r  ); img4=img4(:);
img5 = img(1+r-r:h-r-r,1+r-r:w-r-r); img5=img5(:);
img6 = img(1+r  :h-r  ,1+r-r:w-r-r); img6=img6(:);
img7 = img(1+r+r:h-r+r,1+r-r:w-r-r); img7=img7(:);
img8 = img(1+r+r:h-r+r,1+r  :w-r  ); img8=img8(:);
img9 = img(1+r+r:h-r+r,1+r+r:w-r+r); img9=img9(:);


ff = zeros((h-2*r)*(w-2*r),dim);

% 0th order
ff(:,1) = img1;

% 1st order
ff(:,2) = img1.*img1;
ff(:,3) = img1.*img2;
ff(:,4) = img1.*img3;
ff(:,5) = img1.*img4;
ff(:,6) = img1.*img5;

% 2nd order
ff(:,7)  = ff(:,2).*img1;
ff(:,8)  = ff(:,2).*img2;
ff(:,9)  = ff(:,2).*img3;
ff(:,10) = ff(:,2).*img4;
ff(:,11) = ff(:,2).*img5;

ff(:,12) = ff(:,3).*img2;
ff(:,13) = ff(:,4).*img3;
ff(:,14) = ff(:,5).*img4;
ff(:,15) = ff(:,6).*img5;

ff(:,16) = ff(:,3).*img6;
ff(:,17) = ff(:,4).*img7;
ff(:,18) = ff(:,5).*img8;
ff(:,19) = ff(:,6).*img9;
ff(:,20) = ff(:,4).*img6;
ff(:,21) = ff(:,5).*img7;
ff(:,22) = ff(:,6).*img8;
ff(:,23) = img1.*img6.*img9;
ff(:,24) = ff(:,3).*img7;
ff(:,25) = ff(:,4).*img8;
ff(:,26) = ff(:,5).*img9;
ff(:,27) = ff(:,6).*img2;
ff(:,28) = ff(:,5).*img6;
ff(:,29) = ff(:,6).*img7;
ff(:,30) = img1.*img6.*img8;
ff(:,31) = img1.*img7.*img9;
ff(:,32) = ff(:,3).*img8;
ff(:,33) = ff(:,4).*img9;
ff(:,34) = ff(:,3).*img4;
ff(:,35) = ff(:,4).*img5;

f = sum(ff, 1)';

if nargout >= 2
    ff = reshape(ff',dim,h-2*r,w-2*r);    
end


end
