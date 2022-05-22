function inputdata=preprocess_cnn2(com_data,expand_num)
inputdata = zeros(expand_num,expand_num,2);
com_data=fft2c(com_data);
data = rot90(flip(com_data,2),1);

%zero-padding to 256
[w,h] = size(data);
expand_2D_complex = zeros(expand_num, expand_num) + 1.0i * zeros(expand_num, expand_num);
expand_2D_complex(round((expand_num-w)/2)+1:round((expand_num+w)/2),round((expand_num-h)/2)+1:round((expand_num+h)/2),:)=data;

%ifft
result_2D_complex=ifft2c(expand_2D_complex);

%norm
cur_amp= max(max(abs(result_2D_complex)));
result_2D_complex_norm=result_2D_complex/cur_amp;

%split the real and imag
result_2D_complex_norm=flip(flip(result_2D_complex_norm,1),2);
inputdata(:,:,1)=real(result_2D_complex_norm);
inputdata(:,:,2)=imag(result_2D_complex_norm);
inputdata=single(inputdata);

end