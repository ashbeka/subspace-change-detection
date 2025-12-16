function [F] = cvtLBP_LAC_R(img, s, r, config, option)

rotmap = [1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;...
          5;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;...
          9;32;33;34;24;35;36;37;38;39;40;41;42;43;44;45;...
          13;46;47;48;28;49;50;51;42;52;53;54;55;56;57;58;...
          2;59;60;61;17;62;63;64;32;65;66;67;46;68;69;70;...
          6;62;71;72;21;73;74;75;35;76;77;78;49;79;80;81;...
          10;65;82;83;25;76;84;85;39;86;87;88;52;89;90;91;...
          14;68;92;93;29;79;94;95;43;89;96;97;56;98;99;100;...
          3;60;101;102;18;71;103;104;33;82;105;106;47;92;107;108;...
          7;63;103;109;22;74;110;111;36;84;112;113;50;94;114;115;...
          11;66;105;116;26;77;112;117;40;87;118;119;53;96;120;121;...
          15;69;107;122;30;80;114;123;44;90;120;124;57;99;125;126;...
          4;61;102;127;19;72;109;128;34;83;116;129;48;93;122;130;...
          8;64;104;128;23;75;111;131;37;85;117;132;51;95;123;133;...
          12;67;106;129;27;78;113;132;41;88;119;134;54;97;124;135;...
          16;70;108;130;31;81;115;133;45;91;121;135;58;100;126;136;];
dim = max(rotmap);

if ~exist('config', 'var')
    config = 1;
end
if ~exist('option', 'var')
    option = 'sum';
end

if config == 1, nnpos = [2 4 6 8; 3 5 7 1; 4 6 8 2; 5 7 1 3]';
else            nnpos = [1 3 5 7; 2 4 6 8; 3 5 7 1; 4 6 8 2]';   end
sq2 = 1/sqrt(2);
ssq = ceil(s*sq2);
config_vec = ceil(r*[0,1;sq2,sq2;1,0;-sq2,sq2;])';
nConfig = size(config_vec,2);

Z = double(img);
[h w] = size(Z);

C = Z(1+s:h-s,1+s:w-s);
X = zeros(8,h-2*s,w-2*s);
X(1,:,:) = Z(1+s-ssq:h-s-ssq,1+s-ssq:w-s-ssq)-C;
X(2,:,:) = Z(1+s-s  :h-s-s  ,1+s    :w-s    )-C;
X(3,:,:) = Z(1+s-ssq:h-s-ssq,1+s+ssq:w-s+ssq)-C;
X(4,:,:) = Z(1+s    :h-s    ,1+s+s  :w-s+s  )-C;
X(5,:,:) = Z(1+s+ssq:h-s+ssq,1+s+ssq:w-s+ssq)-C;
X(6,:,:) = Z(1+s+s  :h-s+s  ,1+s    :w-s    )-C;
X(7,:,:) = Z(1+s+ssq:h-s+ssq,1+s-ssq:w-s-ssq)-C;
X(8,:,:) = Z(1+s    :h-s    ,1+s-s  :w-s-s  )-C;
X = double(X>0);

Ftmp = zeros(16*16,1);
for ci = 1:nConfig
    A = reshape([1 2 4 8]*X(nnpos(:,ci),:), h-2*s, w-2*s)+1;
    [hh ww] = size(A);
    rr = config_vec(:,ci);

    D = (A(1+r:hh-r, 1+r:ww-r) - 1) * 16;
    Y = A(1+r+rr(1):hh-r+rr(1),1+r+rr(2):ww-r+rr(2)) + D;
    Ftmp = Ftmp + hist(Y(:), 1:(16*16))';
end

if strcmp(option,'sum')
    F = zeros(dim,1);
    for id = 1:16*16
        F(rotmap(id)) = F(rotmap(id))+Ftmp(id);
    end
elseif strcmp(option,'min')
    F = Inf(dim,1);
    for id = 1:16*16
        F(rotmap(id)) = min(F(rotmap(id)),Ftmp(id));
    end
    F(F==Inf) = 0;
end


% 
% %% ‚P‚W‚O“x‰ñ“]‚µ‚Ä“¯‚¶‚Æ‚±‚ë‚É‚©‚Ô‚Á‚Ä‚é‚æƒtƒ‰ƒO‚ð—§‚Ä‚é
%     
% % nB = 4;   %8;
% % nP = 2^nB;
% % 
% % flag = zeros(nP,nP);
% % for pi = 1:nP
% % for pj = 1:nP
% %     if(flag(pj,pi) == 0)
% %         b1 = dec2bin(pj-1,nB);
% %         b2 = dec2bin(pi-1,nB);
% %         index = pj+(pi-1)*nP;
% %         flag(bin2dec([b2(nB/2+1:end) b2(1:nB/2)])+1,...
% %              bin2dec([b1(nB/2+1:end) b1(1:nB/2)])+1) = index;
% %         flag(pj,pi) = index;
% %     end
% % end
% % end
% % 
% % [~,~,rotmap] = unique(flag(:), 'first');
