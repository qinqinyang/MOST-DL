% Process the outputs of MRiLab software for deep learning
% If you have any questions, please contact the author. (qqyang@stu.xmu.edu.cn)

clc
clear
addpath('tools');
dirname='./scan_from_MRiLab/';
outputdir = './train_data/';

if exist(outputdir)==0
    mkdir(outputdir);
end

phase_num=128;
fre_num = 128;
expand_num = 256;
model_num=512;

order = 1;
fid_dir_all=dir([dirname,'*.mat']);       %list all directories

for loopi = 1:length(fid_dir_all)
    output = zeros(5,expand_num,expand_num);
    fid_dir =[dirname,fid_dir_all(loopi).name];
    load(fid_dir);
    
    % T2
    T2STAR = VSig.T2;
    T2STAR = abs(imresize(T2STAR,[expand_num,expand_num],'nearest'));
    T2 = T2STAR;
    
    % B1
    B1 = VSig.B1;
    B1 = abs(imresize(B1,[expand_num,expand_num],'nearest'));
    B1 = B1;
    
    % M0
    M0 = VSig.Rho;
    M0 = abs(imresize(M0,[expand_num,expand_num],'nearest'));
    M0 = M0;
    mask1 = M0>0;
    
    % input
    Sx = squeeze(VSig.Sx);
    Sy = squeeze(VSig.Sy);
    
    K1 = Sx(1:end)+1i*Sy(1:end);
    K1 = reshape(K1,fre_num,phase_num);
    K1(:,2:2:end) = flipud(K1(:,2:2:end));
    k1_image = ifft2c(K1);
    
    % noise
    k1_image = k1_image/max(abs(k1_image(:)));
    sigma=rand()*0.05;
    both_img_noise=normrnd(0,sigma,128,128)+1.0i*normrnd(0,sigma,128,128);
    K1 = fft2c(k1_image+both_img_noise);
    
    % zero-padding to 256
    expand_K1 = zeros(expand_num, expand_num) + 1.0i * zeros(expand_num, expand_num);
    expand_K1(round((expand_num-fre_num)/2)+1:round((expand_num+fre_num)/2),round((expand_num-phase_num)/2)+1:round((expand_num+phase_num)/2),:)=K1;
    I1=ifft2c(expand_K1);
    
    % norm
    max4norm = max(abs(I1(:)));
    I1 = I1/max4norm;
    I1 = flip(I1,2);
    I1 = flip(I1,1);
    
    input=zeros(expand_num,expand_num,2);
    input(:,:,1) = real(I1);
    input(:,:,2) = imag(I1);
    input = single(input);
    
    t2 = single(T2);
    m0 = single(M0);
    b1 = single(B1.*mask1);
    
    filename=[outputdir,'nomotion_',num2str(order),'.mat'];
    save(filename,'input','t2','b1','m0')
    disp(order);
    order = order+1;
end



