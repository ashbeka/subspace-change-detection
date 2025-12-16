function L = cvtLabel(nNum, varargin)


if nargin <1
       error('error');
end
if nargin == 1
    nClass = 1;
    nSet = 1;
end

if nargin == 2
    nClass = varargin{1};
    nSet = 1;
end

if nargin == 3
    nClass = varargin{1};
    nSet = varargin{2};
end

if nargin > 3
       error('error');
end

L = repmat(cvtVec(repmat(1:nClass,nNum,1))',1,nSet);
