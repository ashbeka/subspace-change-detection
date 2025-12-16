% function [f] = cvtHOG(img)
% 
% if ~isa(img, 'double')
%     img = im2double(img);
% end
% 
% bin_num = 9;        % set here the number of histogram bins
% 
% grad_x = imfilter(img, [-1,0,1;-2,0,2;-1,0,1]);
% grad_y = imfilter(img, [1,2,1;0,0,0;-1,-2,-1]);
% 
% grad_x = grad_x(:);
% grad_y = grad_y(:);
% 
% angles = atan2(grad_y, grad_x);
% angles(angles<0) = angles(angles<0) + pi;
% magnit = ((grad_y.^2).*(grad_x.^2)).^.5;
% bin_index = ceil(angles*bin_num/pi);
% 
% f = zeros(bin_num, 1);
% for iBin = 1:bin_num
%     f(iBin) = sum(magnit(bin_index == iBin));
% end

function [hog] = cvtHOG(img)

if ~isa(img, 'double')
    img = im2double(img);
end

cell_w = 3;
cell_h = 3;
block_w = 3;         %set here the number of HOG windows per bound box
block_h = 3;

b = 9;              %set here the number of histogram bins
[h w] = size(img);  % L num of lines ; C num of columns

cells_w = floor(w/cell_w);
cells_h = floor(h/cell_h);
w = cell_w * cells_w;
h = cell_h * cells_h;
img = img(1:h, 1:w);
grad_xr = imfilter(img, [-1,0,1]);
grad_yu = imfilter(img, [1,0,-1]);
angles = mod(atan2(grad_yu, grad_xr), pi);
magnit = ((grad_yu.^2).*(grad_xr.^2)).^.5;

cells = zeros(b, cells_h, cells_w);
for hi = 1:cells_h
for wi = 1:cells_w
    v_angles = angles(((hi-1)*cell_h+1):(hi*cell_h), ((wi-1)*cell_w+1):(wi*cell_w));
    v_magnit = magnit(((hi-1)*cell_h+1):(hi*cell_h), ((wi-1)*cell_w+1):(wi*cell_w));
    v_angles = v_angles(:);
    v_magnit = v_magnit(:);
    bin_index = mod(round(v_angles*b/pi), b) + 1;
    for bi = 1:b
       cells(bi,hi,wi) = sum(v_magnit(bin_index == bi));
    end
end
end

hog = zeros(cells_h-2, cells_w-2, b*block_h*block_w);
for hi = 1:(cells_h-2)
for wi = 1:(cells_w-2)
    v = cells(:, hi:(hi+block_h-1), wi:(wi+block_w-1));
    hog(hi,wi,:) = cvtNormalize(v(:));
end
end
end
