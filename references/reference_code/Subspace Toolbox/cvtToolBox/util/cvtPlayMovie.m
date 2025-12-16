function cvtPlayMovie(mov, fignum, fps)

wait_time = 1/15;

% draw
if(nargin < 2)
    figure;
elseif(nargin < 3)
    figure(fignum);
    clf(fignum);
else
    figure(fignum);
    clf(fignum);
    wait_time = 1/fps;    
end

m_size =size(mov);

% color
if 4 == numel(m_size)
    for i = 1:m_size(4)
        img = mov(:,:,:,i);
        imshow(img);
        drawnow;
        pause(wait_time);
    end
end

%gray
if 3 == numel(m_size)
    for i = 1:m_size(3)
        img = mov(:,:,i);
%         imshow(img);
        imagesc(img);
        drawnow;
        pause(wait_time);
    end
end

end
