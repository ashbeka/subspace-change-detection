function data = cvtLoadMul(filename)

fid = fopen(filename,'r');

tmp=fgets(fid,256);
frames = sscanf(tmp, '%ld', 1)


tmp=fgets(fid,256);
para = sscanf(tmp, '%d', 3)
width = para(1)'
height = para(2)'
bytetype = para(3)'

% Although the line including '#' at the head is inputCit is discraed
% without using.
tmp=fgets(fid,256);
length(tmp);
if (length(tmp) > 2) % when with commets added after #
else % when without comments
end

%%
% Note if you deal with the mulfile including HLAC, CHLAC,
% data type will be float or double. In this case, please use 'float'
% instead of 'uchar' or 'uchar*3'.
len = (bytetype / 8) * width * height;
if bytetype/8==1 % mono
   data = fread(fid,[len,frames],'uchar');
elseif bytetype/8==3 % color
   data = fread(fid,[len,frames],'uchar*3');
end
data = squeeze(reshape(data,height,width,bytetype/8,frames));
data = permute(data,[2,1,3]);
fclose(fid);
