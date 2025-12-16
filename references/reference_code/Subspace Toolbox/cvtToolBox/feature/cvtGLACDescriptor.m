function f = cvtGLACDescriptor(img)
% cvtGLACDescriptor
% ++ input ++
%  img : gray image
% ++ output ++
%  f : GLAC features (cells*328dim)
%
bin_num = 8;
cell_w = 10;
cell_h = 10;
[h w] = size(img);

cells_w = floor(w/cell_w);
cells_h = floor(h/cell_h);
w = cell_w * cells_w;
h = cell_h * cells_h;
img = img(1:h, 1:w);

[grad_img_bin, grad_img_norm] = cvtIm2GradientVecImg2(img, bin_num);


f = zeros(328, cells_h, cells_w);

for hi = 1:cells_h
for wi = 1:cells_w
    f(:,hi,wi) = cvtGLAC(grad_img_bin(((hi-1)*cell_h+1):(cell_h*hi), ((wi-1)*cell_w+1):(wi*cell_w),:), ...
                         grad_img_norm(((hi-1)*cell_h+1):(cell_h*hi), ((wi-1)*cell_w+1):(wi*cell_w),:),...
                         1);
end
end

end