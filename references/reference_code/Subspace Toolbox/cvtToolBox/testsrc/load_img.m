% make face.m

person_num = 9;
img_num = 200;
img_size = [40 40];

%%
imgs = zeros(img_size(1), img_size(2), img_num, person_num);
for pi = 1:person_num
for imgi = 1:img_num
    img = imread(sprintf('C:/Users/nosaka/Desktop/face_data/train/%1d_%03d.jpg', pi, imgi-1));
    img = im2double(img);
    img = imresize(img, img_size);
    img = histeq(img);
    imgs(:,:,imgi, pi) = img;
    imshow(img);
    drawnow;
end
end

save('face.mat', 'imgs');


%% 
imgs = zeros(img_size(1), img_size(2), img_num, person_num);
for pi = 1:person_num
for imgi = 1:img_num
    img = imread(sprintf('C:/Users/nosaka/Desktop/face_data/test/%1d_%03d.jpg', pi, imgi-1));
    img = im2double(img);
    img = imresize(img, img_size);
    img = histeq(img);
    imgs(:,:,imgi, pi) = img;
    imshow(img);
    drawnow;
end
end

save('face2.mat', 'imgs');
