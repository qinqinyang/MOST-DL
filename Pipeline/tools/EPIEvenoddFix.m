function kout=EPIEvenoddFix(kraw,kpc,ROdim,PEdim,echoi)
if mod(echoi,2)==1
    dir=1;
else
    dir=-1;
end

kout=kraw;
kraw=kraw(:,:,2:2:end);
krawsize=size(kraw); %[128,20,24]
krawindex=cell(1,length(krawsize));
krawindex(:)={':'};

kraw_evenindex=krawindex;
kpc_oddindex=krawindex;
kpc_evenindex=krawindex;

kraw_evenindex(PEdim)={2:2:krawsize(PEdim)};
kpc_oddindex(PEdim)={2};
kpc_evenindex(PEdim)={1};

im_odd=fftk2r(kpc(kpc_oddindex{:}),[],ROdim);
im_even=fftk2r(kpc(kpc_evenindex{:}),[],ROdim);

mask = GetMagBasedMask(abs(im_odd), abs(im_even), 0.1);
ph=abs(mask).*exp(dir*-1i*angle(conj(im_even).*im_odd));

repmatsize=ones(1,length(krawsize));
repmatsize(PEdim)=length(kraw_evenindex{PEdim});
ph=repmat(ph,repmatsize);

im=fftk2r(kraw,[],ROdim);
im(kraw_evenindex{:})=im(kraw_evenindex{:}).*ph;
kout(:,:,2:2:end)=fftr2k(im,[],ROdim);

function [Mask] = GetMagBasedMask(Imgs1, Imgs2, Std2NoiseThreshFactor)

  % Store original shape of Imgs1 (and Imgs2)
  SizeImgsOrig = size(Imgs1) ;
  
  % 3D arrays of magnitude of images. 3D is easier to analyze.
  AbsImgs1 = abs(Imgs1(:, :, :)) ;
  AbsImgs2 = abs(Imgs2(:, :, :)) ;
  
  % Store new shape\size of 3D arrays.
  SizeImgs3D = size(AbsImgs1) ;
  % Make sure it has 3 components. If the 3rd dimension is 1, SizeImgs3D
  %  will only be of length 2, not 3.
  SizeImgs3D(3) = size(AbsImgs1, 3) ;
  
  
  % Find standard deviation between absolute value of Imgs1 and that of
  %  Imgs2. Finds std, per image. Does not take into account that std
  %  analysis should be different at regions with signal (~Gaussian noise)
  %  to regions without signal (~Rician noise).
  StdPerImg = std( reshape( AbsImgs2 - AbsImgs1, ...
                            [], SizeImgs3D(3) ), ...
                   0, 1) ;
  % Determine noise threshold per image, to be a multiple of the std (per
  %  image).
  NoiseThreshPerImg = StdPerImg * Std2NoiseThreshFactor ;
  
  NoiseThreshPerPixel = repmat(reshape(NoiseThreshPerImg, 1, 1, []), ...
                               SizeImgs3D(1:2)) ;
                             
  Mask = (AbsImgs1 > NoiseThreshPerPixel & ...
          AbsImgs2 > NoiseThreshPerPixel) ;
        
  % "Restore" shape of Mask, to original shape of the Imgs:
  Mask = reshape(Mask, SizeImgsOrig) ;

return ;