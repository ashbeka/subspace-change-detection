function [Z,U,D,C] = orzPCA(X,Y,varargin)
%function [Z,U,D,C] = orzPCA(X,Y,varargin)
% 主成分分析 ver.1.00 by ohkawa
% input
%  X: 入力マトリックス
%  Y: 部分空間の次元もしくは寄与率
%  Yが1.0以下ならば， Yは寄与率と判断
%  Yが1.0超過ならば， Yは部分空間の次元と判断
%  第三引数：デフォルトでは，共分散行列のPCA
%  第三引数が'R'の時，自己相関行列のPCA
%  
%　nDim< nNumの時は通常のPCA
%  nDim>=nNumのときは線形カーネルPCA（双対問題）を適用する
% output
%  Z: 主成分空間に射影されたデータ
%  U: 主成分ベクトル
%  J: 固有値
%  C: 寄与率


X = X(:,:);
[nDim,nNum] =size(X);

flgM = true;
if nargin == 3
    if varargin{1} == 'R'
        flgM = false;
    end
end

if Y <= 1 % 寄与率
    cRate = Y;
    if nDim<nNum
        if flgM==true
            C = cov(X',1);
        else
            C = X*X'/nNum;
        end
        [U,tmpD]= eig(C);
        [D ind]= sort(diag(tmpD),'descend');
        U = U(:,ind);
        nSubDim = find(cumsum(D)/sum(D)>=cRate, 1 );
        U=U(:,1:nSubDim);
        D = D(1:nSubDim);
        C = sum(D);
    else
        if flgM==true
            K = X'*X;
            IN =  ones(nNum,nNum)/nNum;
            K = K - IN*K - K*IN + IN*K*IN;
        else
            K = X'*X;
        end
        [A B] = eig(K);
        [B ind] = sort(diag(B),'descend');
        A=A(:,ind);
        D = B/nNum;
        nSubDim = find(cumsum(D)/sum(D)>=cRate, 1 );
        A=A(:,1:nSubDim);
        B=B(1:nSubDim);
        A = A/sqrt(diag(B));
        U = X*A;
        D = D(1:nSubDim);
        C = sum(D);        
    end
elseif Y > 1 % 主成分空間の次元
    nSubDim = floor(Y);
    if nDim<nNum
        if flgM==true % 共分散行列
            C = cov(X',1);
        else % 自己相関行列
            C = X*X'/nNum;
        end
        
        OPTS.disp = 0;
        [U tmpD] = eigs(C,nSubDim,'lm',OPTS); %[U,tmpD]= eigs(C,nSubDim);
        [D ind]= sort(diag(tmpD),'descend');
        U = U(:,ind);
        C = sum(D);
    else %カーネル
        if flgM==true % 共分散行列
            K = X'*X;
            IN =  ones(nNum,nNum)/nNum;
            K = K - IN*K - K*IN + IN*K*IN;
        else  % 自己相関行列
            K = X'*X;
        end
        OPTS.disp = 0;        
        [A B] = eigs(K,nSubDim,'lm',OPTS);
        [B ind] = sort(diag(B),'descend');
        A=A(:,ind);
        D = B/nNum;
        A = A/sqrt(diag(B));
        U = X*A;
        C = sum(D);
    end
end
Z = U'*X;
