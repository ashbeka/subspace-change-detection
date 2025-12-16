function [A,D,C,K] = orzKPCA(X,Y,nSigma,varargin)
% ƒJ?[ƒlƒ‹Žå?¬•ª•ª?Í ver.1.00 by ohkawa
% input
%  X: “ü—Íƒ}ƒgƒŠƒbƒNƒX
%  Y: •”•ª‹óŠÔ‚ÌŽŸŒ³‚à‚µ‚­‚ÍŠñ—^—¦
%  Y‚ª1.0ˆÈ‰º‚È‚ç‚Î?C Y‚ÍŠñ—^—¦‚Æ”»’f
%  Y‚ª1.0’´‰ß‚È‚ç‚Î?C Y‚Í•”•ª‹óŠÔ‚ÌŽŸŒ³‚Æ”»’f
%  nSigma: ƒJ?[ƒlƒ‹ƒpƒ‰ƒ??[ƒ^?iGaussian Kernel?j
%  ‘æŽOˆø?”?FƒfƒtƒHƒ‹ƒg‚Å‚Í?C‹¤•ªŽU?s—ñ‚ÌPCA
%  ‘æŽOˆø?”‚ª'R'‚ÌŽž?CŽ©ŒÈ‘ŠŠÖ?s—ñ‚ÌPCA
%
% output
%  A: Žå?¬•ªƒxƒNƒgƒ‹‚ÌŒ‹?‡ŒW?”
%  D: ŒÅ—L’l
%  C: Šñ—^—¦
%  K: ƒJ?[ƒlƒ‹ƒOƒ‰ƒ€ƒ}ƒgƒŠƒbƒNƒX


X = X(:,:);
[nDim,nNum] =size(X);

flgM = true;
if nargin == 4
    if varargin{1} == 'R'
        flgM = false;
    end
end

K=exp(-orzL2Distance(X,X)/nSigma);
if flgM==true % ‹¤•ªŽU?s—ñ
    IN =  ones(nNum,nNum)/nNum;
    K = K - IN*K - K*IN + IN*K*IN;
end


if Y <= 1 % Šñ—^—¦
    cRate = Y;
    [A B] = eig(K);
    [B ind] = sort(diag(B),'descend');
    A=A(:,ind);
    D = B/nNum;
    nSubDim = find(cumsum(D)/sum(D)>=cRate, 1 );
    A=A(:,1:nSubDim);
    B=B(1:nSubDim);
    A = A/sqrt(diag(B));
    %U = X*A;
    D = D(1:nSubDim);
    C = sum(D);
    
elseif Y > 1 % Žå?¬•ª‹óŠÔ‚ÌŽŸŒ³
    nSubDim = floor(Y);
    OPTS.disp = 0;
    [A B] = eigs(K,nSubDim,'lm',OPTS);  %[A B] = eigs(K,nSubDim);
    %[A B] = eigs(K,nSubDim);
    [B ind] = sort(diag(B),'descend');
    A=A(:,ind);
    D = B/nNum;
    A = A/sqrt(diag(B));
    C = sum(D);
end
