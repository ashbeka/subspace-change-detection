function [Vimg] = cvtIm2HueVecImg(img)

img = rgb2hsv(img);
hue = img(:,:,1) * 2 * pi;
r = img(:,:,2) .* img(:,:,3);
Vimg = zeros([size(hue) 3]);
Vimg(:, :, 1) = r.*cos(hue);
Vimg(:, :, 2) = r.*sin(hue);

% img = rgb2hsv(img);
% hue = img(:,:,1) * 2 * pi;
% Vimg = zeros([size(hue) 3]);
% Vimg(:, :, 1) = cos(hue);
% Vimg(:, :, 2) = sin(hue);
% 

end
