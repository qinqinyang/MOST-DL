function complex_data = tensor2complex(data,w,h,n)
complex_data=zeros(w,h,n);
for i=1:n
    re=reshape(data(:,:,2*i-1),[w,h]);
    im=reshape(data(:,:,2*i),[w,h]);
    complex_data(:,:,i)=re+1.0i*im;
end
end