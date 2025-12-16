% test yale
clear;
load('yaleface_64x64');

imgs = imgs/255;
s = size(imgs);

flags = sum(reshape(imgs,s(1),s(2),1, s(3)),1) == 0;
flags = squeeze(flags);

imshow(flags);

flags(flags==1) = 0;

imagesc(flags);

% reject_subset = ceil(find(flags==1)/64);
% reject_index = mod(find(flags==1),64)+1;
% 
% flags(reject_subset, reject_index) = 0;
% imagesc(flags);


