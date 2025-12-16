function [ output_args ] = cvtSaveMul( filename, data )
%SAVEMUL この関数の概要をここに記述
%   詳細説明をここに記述

fid = fopen(filename,'w');
[h,w,n] = size(data);

fprintf(fid,'%d\n%d %d 8\n#\n',n,h,w);
fwrite(fid,permute(data,[2,1,3]),'uint8');
fclose(fid);
