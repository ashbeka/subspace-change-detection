function [img_Re img_Im] = cvtGaborFilter(img, kv, sigma, phi)

if ~isa(img, 'double')
   img = im2double(img); 
end

if ~exist('kv', 'var'),   kv = pi / (2*sqrt(2)); end;
if ~exist('sigma', 'var'),sigma = pi; end
if ~exist('phi', 'var'),  phi = 0; end

[filter_re filter_im] = cvtGabor_fn(kv, sigma, phi);

img_Re = imfilter(img, filter_re, 'symmetric');
img_Im = imfilter(img, filter_im, 'symmetric');

end

% function [img_Re img_Im] = cvtGaborFilter(img, bw, gamma, psi, lambda, theta)
% % bw    = bandwidth, (1)
% % gamma = aspect ratio, (0.5)
% % psi   = phase shift, (0)
% % lambda= wave length, (>=2)
% % theta = angle in rad, [0 pi)
% % http://matlabserver.cs.rug.nl/edgedetectionweb/web/edgedetection_params.html 
% 
% if ~isa(img, 'double')
%    img = im2double(img); 
% end
% 
% if ~exist('bw', 'var'),     bw = 1; end;
% if ~exist('psi', 'var'),    psi = 0; end
% if ~exist('gamma', 'var'),  gamma = 0.5; end
% if ~exist('lambda', 'var'), lambda  = 8; end;
% if ~exist('theta', 'var'),  theta   = 0; end;
% 
% filter_re = cvtGabor_fn(bw,gamma,psi     ,lambda,theta);
% filter_im = cvtGabor_fn(bw,gamma,psi+pi/2,lambda,theta);
% 
% img_Re = imfilter(img, filter_re, 'symmetric');
% img_Im = imfilter(img, filter_im, 'symmetric');
% 
% end
