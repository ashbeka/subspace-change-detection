function [gb_re, gb_im] = cvtGabor_fn(kv, sigma, phi)

sz = fix(4*sigma);

[x y] = meshgrid(-fix(sz/2):fix(sz/2),fix(sz/2):-1:fix(-sz/2));
x_theta =  x*cos(phi) + y*sin(phi);
y_theta = -x*sin(phi) + y*cos(phi);
gb = kv*kv/(sigma*sigma)*exp(-0.5*kv*kv/sigma^2*(x_theta.^2 + y_theta.^2));
gb_re = gb.*(cos(kv*x_theta) - exp(-sigma*sigma*0.5));
gb_im = gb.*(sin(kv*x_theta));

end
% function gb = cvtGabor_fn(bw, gamma, psi, lambda, theta)
% % bw    = bandwidth, (1)
% % gamma = aspect ratio, (0.5)
% % psi   = phase shift, (0)
% % lambda= wave length, (>=2)
% % theta = angle in rad, [0 pi)
%  
% sigma = lambda/pi * sqrt(log(2)/2) * (2^bw+1)/(2^bw-1);
% sigma_x = sigma;
% sigma_y = sigma/gamma;
% 
% sz=fix(8*max(sigma_y,sigma_x));
% if mod(sz,2)==0, sz=sz+1;end
% 
% % alternatively, use a fixed size
% % sz = 60;
%  
% [x y] = meshgrid(-fix(sz/2):fix(sz/2),fix(sz/2):-1:fix(-sz/2));
% % x (right +)
% % y (up +)
% 
% % Rotation
% x_theta =  x*cos(theta) + y*sin(theta);
% y_theta = -x*sin(theta) + y*cos(theta);
%  
% gb = exp(-0.5*(x_theta.^2/sigma_x^2 + y_theta.^2/sigma_y^2)) ...
%       .*cos(2*pi*x_theta/lambda+psi);
% 
% end
