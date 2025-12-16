load('yaleface_88x88.mat');
img = imgs(:,:,1);


%% 
[imgg_re imgg_im] = cvtGaborFilter(img, pi/(2*sqrt(2)), pi, 0*pi/8);

figure(1); imshow(img);
figure(2); imshow(imgg_re);
figure(3); imagesc(imgg_re);
figure(4); imagesc(abs(imgg_re));
figure(5); imshow(imgg_im);
figure(6); imagesc(imgg_im);
figure(7); imagesc(abs(imgg_im));
