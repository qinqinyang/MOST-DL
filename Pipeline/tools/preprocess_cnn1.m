function [under,mask]=preprocess_cnn1(ksp_cal,w,h,coiln)
scale = 6000;
under=zeros(w,h,coiln*2);
temp_no=zeros(w,h,coiln);
temp_no(:,2:end-1,:) = ksp_cal;
temp_no = temp_no * scale;
temp_no(:,2:2:end,:)=0+1.0i*0;
jj=1;
for i=1:coiln
    temp = (reshape(temp_no(:,:,i),[w,h]));
    under(:,:,jj)=real(temp);
    jj=jj+1;
    under(:,:,jj)=imag(temp);
    jj=jj+1;
end

% mask
mask=ones(w,h,coiln*2);
mask(:,2:2:end,:)=0+1.0i*0;
mask(:,1,:)=0+1.0i*0;

under=single(under);
mask=single(mask);

end