% Process the outputs of SPROM software for deep learning
% If you have any questions, please contact the author. (qqyang@stu.xmu.edu.cn)

clc
clear
addpath('tools');
fid_dir ='./scan_from_SPROM/';
outputdir = './train_data/';

if exist(outputdir)==0
    mkdir(outputdir);
end

phase_num=128;
fre_num = 128;
expand_num = 256;
model_num_x=512;
model_num_z=512;
order=1 ;
Fov=0.22;
output=zeros(expand_num,expand_num,6);

fid_file_all1=dir([fid_dir,'Team*.B1']);    %list all files
fid_file_all2=dir([fid_dir,'tempd*']);    %list all files
fid_file_all3=dir([fid_dir,'Team*.m0']);    %list all files
fid_file_all4=dir([fid_dir,'Team*.T2']);    %list all files
fid_file_all7=dir([fid_dir,'*.vx']);    %list all files
fid_file_all8=dir([fid_dir,'*.vz']);    %list all files
fid_file_all9=dir([fid_dir,'*.rot']);    %list all files

for loopj = 1:length(fid_file_all2)
    
    %m0
    fid_file =[fid_dir,fid_file_all3(loopj).name];
    fip_dif=fopen(fid_file,'rb');
    [Array_2D_dif,num]=fread(fip_dif,inf,'double');
    data_temp=Array_2D_dif(:,:);
    data_temp=reshape(data_temp,model_num_x,model_num_z);
    fclose(fip_dif);
    M0=imresize(data_temp,[expand_num,expand_num],'nearest');
    
    MASK=zeros(size(M0));
    MASK(M0>0)=1;
    
    %B1
    fid_file =[fid_dir,fid_file_all1(loopj).name];
    fip_dif=fopen(fid_file,'rb');
    [Array_2D_dif,num]=fread(fip_dif,inf,'double');
    data_temp=Array_2D_dif(:,:);
    data_temp=reshape(data_temp,model_num_x,model_num_z);
    fclose(fip_dif);
    B1=imresize(data_temp,[expand_num,expand_num],'nearest');
    B1 = B1.*MASK;
    
    %T2
    fid_file =[fid_dir,fid_file_all4(loopj).name];
    fip_dif=fopen(fid_file,'rb');
    [Array_2D_dif,num]=fread(fip_dif,inf,'double');
    data_temp=Array_2D_dif(:,:);
    origin_2D=reshape(data_temp,model_num_x,model_num_z);
    fclose(fip_dif);
    T2=imresize(origin_2D,[expand_num,expand_num],'nearest');
    T2 = T2.*MASK;
    
    % vx
    fid_file =[fid_dir,fid_file_all7(loopj).name];
    fip_dif=fopen(fid_file,'rb');
    [Array_2D_dif,num]=fread(fip_dif,inf,'double');
    data_temp=Array_2D_dif(:,:);
    data_temp=reshape(data_temp,model_num_x,model_num_z);
    fclose(fip_dif);
    Vx=imresize(data_temp,[expand_num,expand_num],'nearest');
    Vx = Vx.*MASK;
    
    % vz
    fid_file =[fid_dir,fid_file_all8(loopj).name];
    fip_dif=fopen(fid_file,'rb');
    [Array_2D_dif,num]=fread(fip_dif,inf,'double');
    data_temp=Array_2D_dif(:,:);
    data_temp=reshape(data_temp,model_num_x,model_num_z);
    fclose(fip_dif);
    Vz=imresize(data_temp,[expand_num,expand_num],'nearest');
    Vz = Vz.*MASK;
    
    % rot
    fid_file =[fid_dir,fid_file_all9(loopj).name];
    fip_dif=fopen(fid_file,'rb');
    [Array_2D_dif,num]=fread(fip_dif,inf,'double');
    data_temp=Array_2D_dif(:,:);
    data_temp=reshape(data_temp,model_num_x,model_num_z);
    fclose(fip_dif);
    Rot=imresize(data_temp,[expand_num,expand_num],'nearest');
    Rot = Rot.*MASK;
    
    % MOLED image
    fid_file = [fid_dir,fid_file_all2(loopj).name];
    origin_1D_data=load(fid_file, '-ascii');
    data_size=size(origin_1D_data);
    origin_1D_complex=origin_1D_data(:,1)+1.0i*origin_1D_data(:,2);
    
    origin_2D_complex=reshape(origin_1D_complex,[fre_num,phase_num]);
    
    % EPI shift
    for j = 2 : 2 : phase_num
         origin_2D_complex(:,j) = flipud(origin_2D_complex(:,j));
    end
    k1_image = ifft2c(origin_2D_complex);
    
    % noise
    k1_image = k1_image/max(abs(k1_image(:)));
    sigma=rand()*0.05;
    both_img_noise=normrnd(0,sigma,128,128)+1.0i*normrnd(0,sigma,128,128);
    K1 = fft2c(k1_image+both_img_noise);
    
    % zero-padding to 256
    expand_2D_complex = zeros(expand_num, expand_num) + 1.0i * zeros(expand_num, expand_num);
    expand_2D_complex(round((expand_num-fre_num)/2)+1:round((expand_num+fre_num)/2),floor((expand_num-phase_num)/2)+1:floor((expand_num+phase_num)/2))=K1;
    both_image_2D=ifft2c(expand_2D_complex);
    
    % norm
    cur_amp = max(abs(both_image_2D(:)));
    both_image_2D_norm=both_image_2D/cur_amp;
    
    % input
    input=zeros(expand_num,expand_num,2);
    input(:,:,1)=fliplr(flipud((real(both_image_2D_norm))));
    input(:,:,2)=fliplr(flipud((imag(both_image_2D_norm))));
    input = single(input);
    
    % T2
    t2=single(fliplr(flipud(T2)));
    
    % B1
    b1=single(fliplr(flipud(B1)));
    
    % M0
    m0=single(fliplr(flipud(M0/0.035)));
    
    % Motion field
    Vx=fliplr(rot90(fliplr(flipud(Vx))));
    Vz=fliplr(rot90(fliplr(flipud(Vz))));
    Rot=fliplr(rot90(fliplr(flipud(Rot))));
    for i=1:expand_num
        for j=1:expand_num
            Vx2(i,j)=Vx(i,j)+Rot(i,j)*(i-expand_num/2-1)*Fov/expand_num;
            Vz2(i,j)=Vz(i,j)+Rot(i,j)*(j-expand_num/2-1)*Fov/expand_num;
        end
    end
    Vx2=(fliplr(rot90(Vx2)));Vx2=(Vx2+0.1)/0.1;Vx2=Vx2.*MASK;
    Vz2=(fliplr(rot90(Vz2)));Vz2=(Vz2+0.1)/0.1;Vz2=Vz2.*MASK;
    vx=single(Vx2);
    vy=single(Vz2);
    
    filename=[outputdir,'motion1_',num2str(order),'.mat'];
    save(filename,'input','t2','b1','m0','vx','vy')
    disp(filename)
    order = order+1;
end