function [F] = cvtLBP_R_LAC(img, s, r)

map = [ 1,2,2,3,2,4,3,5,2,6,4,7,3,8,5,9,...
        2,10,6,11,4,12,7,13,3,14,8,15,5,16,9,17,...
        2,6,10,14,6,18,11,19,4,18,12,20,7,21,13,22,...
        3,11,14,23,8,24,15,25,5,19,16,26,9,27,17,28,...
        2,4,6,8,10,12,14,16,6,18,18,21,11,24,19,27,...
        4,12,18,24,12,29,20,30,7,20,21,31,13,30,22,32,...
        3,7,11,15,14,20,23,26,8,21,24,31,15,31,25,33,...
        5,13,19,25,16,30,26,34,9,22,27,33,17,32,28,35,...
        2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,...
        6,14,18,19,18,20,21,22,11,23,24,25,19,26,27,28,...
        4,8,12,16,18,21,24,27,12,24,29,30,20,31,30,32,...
        7,15,20,26,21,31,31,33,13,25,30,34,22,33,32,35,...
        3,5,7,9,11,13,15,17,14,19,20,22,23,25,26,28,...
        8,16,21,27,24,30,31,32,15,26,31,33,25,34,33,35,...
        5,9,13,17,19,22,25,28,16,27,30,32,26,33,34,35,...
        9,17,22,28,27,32,33,35,17,28,32,35,28,35,35,36;];
dim = max(map);

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

% LBP‚ğ‹‚ß‚é
A = [128,64,32,16,8,4,2,1]*X(:,:)+1;

% ‰ñ“]•s•Ï‚É‚·‚é
A = map(A(:));

% % ‹¤‹N‚ğ‹‚ß‚é
% A = reshape(A, h-2*s, w-2*s);
% [hh ww] = size(A);
% D = (A(1+r:hh-r, 1+r:ww-r) - 1) * dim;
% F = zeros(dim*dim, 1);
% for ci = 1:nConfig
%     rr = config_vec(:,ci);
%     Y = A(1+r+rr(1):hh-r+rr(1),1+r+rr(2):ww-r+rr(2)) + D;
%     F = F + hist(Y(:), 1:(dim*(dim)))';
% end
A = reshape(A, h-2*s, w-2*s);
[hh ww] = size(A);
D = (A(1+r:hh-r, 1+r:ww-r) - 1) * dim;
F = zeros(dim*dim, 1);
for ci = 1:nConfig
    rr = config_vec(:,ci);
    Y = A(1+r+rr(1):hh-r+rr(1),1+r+rr(2):ww-r+rr(2)) + D;
    F = F + hist(Y(:), 1:(dim*(dim)))';
end
F = reshape(F,dim,dim);
F = F + tril(F, -1)';
F = F(logical(triu(ones(dim,dim),0)));
F = F(:);

% 
% %% ‚P‚W‚O“x‰ñ“]‚µ‚Ä“¯‚¶‚Æ‚±‚ë‚É‚©‚Ô‚Á‚Ä‚é‚æƒtƒ‰ƒO‚ğ—§‚Ä‚é
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
