function PeakList = cvtFindLocalPeakX(X, Flg, Thres, N )
% PeakList = cvtFindLocalPeak(X, Flg, Thres )
%
% 引数
% X:        データ，マップ
% Flg：      Flg==1：極大点探索，Flg==-1：極小点探索
% Thres：    探索の閾値
%
% 戻り値
% PeakList： 極大点（極小点）のリスト，ソート済み

%%
if nargin < 2
   error('error');
end


if nargin == 2
   B = ones(size(X));
   N = 1;
end

if nargin == 3
   if Flg == 1
      B = (X > (Thres));
   elseif Flg == -1
      B = (X < (Thres));
   else
      error('error');
   end
   N=1;
end

if nargin == 4
   if Flg == 1
      B = (X > (Thres));
   elseif Flg == -1
      B = (X < (Thres));
   else
      error('error');
   end
end

if nargin > 4
   error('error');
end

B(1,:) = 0;
B(end,:) = 0;
B(:,1) = 0;
B(:,end) = 0;


%Y = NaN*zeros(size(X));
[a,b] = find(B);
Candinate = [a,b]';
PeakList = zeros([3,size(Candinate,2)]);
cnt=0;

if Flg == 1
   for l= 1:size(Candinate,2);
      y = Candinate(1,l);
      x =  Candinate(2,l);
      tmp = X(y-1:y+1,x-1:x+1);
      tmp = tmp(:);
      [val ind ] = sort(tmp,'descend');
      % ８近傍の局所最大点のみ

      for n=1:N
         if (ind(n) == 5)
            cnt = cnt+1;
            PeakList(:,cnt) = [Candinate(:,l);val(n)];
         end
      end
   end
   PeakList = PeakList(:,1:cnt);
   [ksk ind] = sort(PeakList(3,:),'descend');
   PeakList = PeakList(:,ind);

elseif Flg == -1
   for l= 1:size(Candinate,2);
      y = Candinate(1,l);
      x =  Candinate(2,l);
      tmp = X(y-1:y+1,x-1:x+1);
      tmp = tmp(:);
      [val ind ] = sort(tmp,'ascend');
      % ８近傍の局所最大点のみ
      for n=1:N
         if (ind(n) == 5)
            cnt = cnt+1;
            PeakList(:,cnt) = [Candinate(:,l);val(n)];
         end
      end
   end
   PeakList = PeakList(:,1:cnt);
   [ksk ind] = sort(PeakList(3,:),'ascend');
   PeakList = PeakList(:,ind);
end

