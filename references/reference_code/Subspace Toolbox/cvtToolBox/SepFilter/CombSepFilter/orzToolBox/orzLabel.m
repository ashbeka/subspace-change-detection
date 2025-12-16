function L = orzLabel(nNum,varargin)


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

A=repmat(1:nClass,[nNum,nSet]);
L=A(:)';

