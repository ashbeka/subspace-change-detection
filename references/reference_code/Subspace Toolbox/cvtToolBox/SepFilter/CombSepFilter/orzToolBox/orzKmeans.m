function [L M Score]= orzKmeans(X,k)
% function [L M Score]= orzKmeans(X,k)
% X:データ
% k:クラスタ数

X = X(:,:);
[nDim nNum] = size(X);

L = Plus2Select(X,k);
M = zeros(nDim,k);

cnt = 0;

while 1
    for I=1:k
        M(:,I) = mean(X(:,L==I),2);
    end
    D = sqrt(orzL2Distance(M,X));
    [val,LN] = min(D,[],1);    
    Score = mean(val);
    
    HC = histc(LN,[1:k]);
    SS = find(HC == 0);
    while sum(HC==0)~=0
        disp('Re')
        cnt = cnt+1;
        if cnt < 10
            LN(ceil(rand(1)*nNum*(1-eps))) = SS(1);
            HC = histc(LN,[1:k]);
            SS = find(HC == 0);
        else
            cnt = 0;
            disp('Plus2Select')
            L = Plus2Select(X,k);
        end
    end    
    if LN == L
        break;
    end
    L=LN;
end
L = LabelSort(L);
for I=1:k
    M(:,I) = mean(X(:,L==I),2);
end

% %%
% while 1
%     for I=1:k
%         M(:,I) = mean(X(:,L==I),2);
%     end
%     D = orzL2Distance(M,X);
%     [val,L_now] = min(D,[],1);
%     Score = mean(sqrt(val));
%     if size(unique(L_now),2) ~= k
%         L_now = Plus2Select(X,k);
%         'Re Initialize'
%     elseif L_now == L
%         break;
%     end
%     L=L_now;
% end
% L = LabelSort(L);
% for I=1:k
%     M(:,I) = mean(X(:,L==I),2);
% end





function L = Plus2Select(X,k)
[nDim nNum] = size(X);
M = zeros(nDim,k);
tmpL = zeros(1,k);
tmpL(1)=ceil(rand(1)*(nNum-eps));
M(:,1) = X(:,tmpL(1));

E = zeros(1,nNum);
for I=1:k-1        
    E = E + orzL2Distance(X(:,tmpL(I)),X);
    E(tmpL(1:I)) = 0;
    S=cumsum(E)/sum(E,2);
    A=(S-rand(1));
    A(A<=0)=Inf;
    A(tmpL(1:I)) = Inf;
    [val ind] = min(A);
    M(:,I+1) = X(:,ind);
    tmpL(I+1) = ind;
end
[val,L] = min(orzL2Distance(M,X));


function K = LabelSort(L)

uN = size(unique(L),2);
cL = L;
C = zeros(1,uN);
for I = 1:uN
    C(I) =cL(1) ;
    cL(cL==cL(1) )=[];
end
K = zeros(size(L));
for I = 1:uN
    K(L==C(I)) = I;
end