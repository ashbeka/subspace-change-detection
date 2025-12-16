function [Y W] = cvtPLS(X, Y, dim)

W = zeros(size(X,1), dim);

Xold = X;

for i = 1:dim
    [W(:,i),~,~] = svds(X*Y',1);
    wx = W(:,i)' * X;
    X = X - X*(wx'*wx)/(wx*wx');
end
    
Y = W'*Xold;


% 
% [X, mux] = cvtCentering(X);
% [Y, muy] = cvtCentering(Y);
% 
% for i = 1:dim
%    YX = Y * X';
%    u(:, i) = YX(:,1)' / norm(YX(:,1));
%    if(size(Y, 1) > 1 % only loop if dimension greater than 1
%        uold = u(:,i) + 1;
%        while norm(u(:,i) - uold) > 0.001
%            uold = u(:,i);
%            tu = YX'  * YX * u(:,i);
%            u(:, i) = tu / norm(tu);
%        end
%    end
%    
%    t = u(:,i)' * X;
%    c(:,i) = Y * t/(t'*t);
%    p(:,i) = X * t/(t'*t);
%    trainY = trainY + 
    
% end
