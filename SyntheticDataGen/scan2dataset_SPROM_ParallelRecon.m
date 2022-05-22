% Single-coil data to multi-coil data for deep learning
% If you have any questions, please contact the author. (qqyang@stu.xmu.edu.cn)

clc
clear
addpath('tools');
dirname='./scan_from_SPROM/';
coil_fn='./coilmaps/';
outputdir = './train_PI/';

if exist(outputdir)==0
    mkdir(outputdir);
end

phase_num=128;
fre_num = 128;
coiln=16;

order= 1;
fid_file_all2=dir([dirname,'tempd_*']);    %list all files
fid_file_coil=dir([coil_fn,'*.mat']);
coil_n=length(fid_file_coil);

for loopj = 1:length(fid_file_all2)
    fid_file = [dirname,fid_file_all2(loopj).name];
    origin_1D_data=load(fid_file, '-ascii');
    data_size=size(origin_1D_data);
    origin_1D_complex=origin_1D_data(:,1)+1.0i*origin_1D_data(:,2);
    origin_2D_complex=reshape(origin_1D_complex,[fre_num,phase_num]);
    
    % EPI shift
    for j = 2 : 2 : phase_num
         origin_2D_complex(:,j) = flipud(origin_2D_complex(:,j));
    end
    im = ifft2c(origin_2D_complex);
    cup=max(abs(im(:)));
    im=im/cup;
    
    % select coil
    idx_coil=randperm(coil_n,1);
    coil_file=[coil_fn,fid_file_coil(idx_coil).name];
    load(coil_file);
    
    im_multi=zeros(fre_num,phase_num,coiln);
    temp=2.5+3*rand();
  
    for ic=1:coiln
        im_multi(:,:,ic)=im.*coil_map(:,:,ic)*temp;
    end
    
    im_multi_noise = zeros(size(im_multi));
    rand_factor=0.015*rand();
    for ic=1:coiln
        s_sigma=rand_factor+0.002*2*(rand()-0.5);
        D2_noise=normrnd(0,s_sigma,128,128)+1.0i*normrnd(0,s_sigma,128,128);
        im_multi_noise(:,:,ic)=im_multi(:,:,ic)+D2_noise;
    end
    
    
    k_multi_temp=fft2c(im_multi_noise);
    kun_multi=zeros(size(im_multi));
    kun_multi(:,3:2:end,:)=k_multi_temp(:,3:2:end,:);
    
    if rand()>0
        k_multi=fft2c(im_multi_noise);
    else
        k_multi=fft2c(im_multi);
    end
    
    under=zeros(fre_num,phase_num,coiln*2);
    label=zeros(fre_num,phase_num,coiln*2);
    
    for zzz=1:coiln
        under(:,:,(zzz*2)-1)=real(kun_multi(:,:,zzz));
        under(:,:,(zzz*2))=imag(kun_multi(:,:,zzz));
        label(:,:,(zzz*2)-1)=real(k_multi(:,:,zzz));
        label(:,:,(zzz*2))=imag(k_multi(:,:,zzz));
    end
    
    mask=ones(fre_num,phase_num,coiln*2);
    mask(:,2:2:end,:)=0+1.0i*0;
    mask(:,1,:)=0+1.0i*0;
    
    nankey1 = isnan(label);
    nankey2 = isnan(under);
    if sum(nankey1(:))==0 && sum(nankey2(:))==0
        disp(order)
        filename=[outputdir,'SPROM_',num2str(order),'.mat'];
        save(filename,'under','label','mask');
        order = order+1;
    end
end