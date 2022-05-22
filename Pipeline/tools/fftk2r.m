function res = fftk2r(x,n,dim)
%% 
% Do Fourier transform in MRI
% usage:  res = ffts(x);
%         res = ffts(x,n);
%         res = ffts(x,n,dim);
% @Zhiyong Zhang, 2016, zhiyongxmu@gmail.com

switch nargin
    case 0
        error('No input found');
    case 1
        res = sqrt(length(x))*fftshift(ifft(ifftshift(x))); 
    case 2
        res = sqrt(size(x,1))*fftshift(ifft(ifftshift(x,1),n,1),1);
    case 3
        res = sqrt(size(x,dim))*fftshift(ifft(ifftshift(x,dim),n,dim),dim);
    otherwise
        error('Too much inputs');
end
