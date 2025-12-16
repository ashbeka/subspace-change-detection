function B = vidResize(A,newsize)
if ~isempty(A)
    B = LinResize(A,newsize);
else
    B=[];
end
end