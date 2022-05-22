function res = fftr2k(x,n,dim)
 
% Do Fourier transform in MRI
% usage:  res = fftk(x);
%         res = fftk(x,n);
%         res = fftk(x,n,dim);
% @Zhiyong Zhang, 2016, zhiyongxmu@gmail.com

switch nargin
    case 0
        error('No input found');
    case 1
        res = 1/sqrt(length(x))*fftshift(fft(ifftshift(x))); 
    case 2
        res = 1/sqrt(size(x,1))*fftshift(fft(ifftshift(x,1),n,1),1);
    case 3
        res = 1/sqrt(size(x,dim))*fftshift(fft(ifftshift(x,dim),n,dim),dim);
    otherwise
        error('Too much inputs');
end
      