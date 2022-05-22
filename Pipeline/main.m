%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example code for evaluating the whole pipeline trained by MOST-DL
% The models (.pth) should be downloaded from xxxxxx
% This code was tested in MATLAB R2019a and Pytorch 1.7.1 (Anaconda python 3.6)
% ksp: under-sampled multi-coil raw data
% knav: navigator data for 3-line linear phase correction
% acs: autocalibration signal (ACS) for calibration-based parallel reconstruction
% If you have any questions, please contact the author. (qqyang@stu.xmu.edu.cn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
clear classes
obj2 = py.importlib.import_module('MOST_DL');
py.importlib.reload(obj2);
addpath('tools');

% load motion-corrupted multi-coil data
load multicoil_data_MOLED.mat

% 3-line linear phase correction
ksp=EPIEvenoddFix(ksp,knav,1,3,1);
ksp_cal = permute(ksp,[1,3,2]);

figure;
imshow3(rot90(abs(ifft2c(ksp_cal)),1),[0 0.0002],[2,8]);
title('Under-sampled multi-coil SE-MOLED data (R=2)');

%% proposed method
% pre-processing for CNN1
w=128;h=128;coiln=16;voxsize=1.72;refcoil=6;radi=12;expand_num=256;
[under,mask] = preprocess_cnn1(ksp_cal,w,h,coiln);
under_np = py.numpy.array(under); % mat to numpy.ndarray
mask_np = py.numpy.array(mask); % mat to numpy.ndarray

% parallel reconstruction using CNN1
multi_mostdl = double(py.MOST_DL.CNN1(under_np, mask_np,'./models/','MOLED_ParallelRec'));
multi_mostdl = tensor2complex(multi_mostdl,w,h,coiln);  % tensor to complex data

figure;
imshow3(rot90(abs((multi_mostdl)),1),[0 1],[2,8]);
title('Reconstructed multi-coil SE-MOLED data using MOST-DL');

% pre-processing for CNN2
[com_mostdl,sen] = adaptive_cmb_2d(multi_mostdl,[voxsize,voxsize],refcoil,radi); % coil combination
input_mostdl = preprocess_cnn2(com_mostdl,expand_num); % zero-padding
input_mostdl_np = py.numpy.array(input_mostdl); % mat to numpy.ndarray

% T2 mapping with MoCo using CNN2
t2_mostdl = double(py.MOST_DL.CNN2(input_mostdl_np,'./models/','MOLED_T2map_MoCo'));

%% conventional method
% parallel reconstruction using GRAPPA
coiln=16;voxsize=1.72;refcoil=6;radi=12;expand_num=256;
multi_grappa=GRAPPA(ksp_cal,acs,[5,5],0.01);  % GRAPPA recon

figure;
imshow3(rot90(abs(ifft2c(multi_grappa)),1),[0 0.0002],[2,8]);
title('Reconstructed multi-coil SE-MOLED data using GRAPPA');

% pre-processing for CNN2
[com_grappa,sen] = adaptive_cmb_2d(ifft2c(multi_grappa),[voxsize,voxsize],refcoil,radi); % coil combination
input_grappa = preprocess_cnn2(com_grappa,expand_num); % zero-padding
input_grappa_np = py.numpy.array(input_grappa); % mat to numpy.ndarray

% T2 mapping w/o MoCo using CNN2+
t2_grappa = double(py.MOST_DL.CNN2(input_grappa_np,'./models/','MOLED_T2map_noMoCo'));

%% results
figure; imshow3(rot90(flip(cat(3,zpad(abs(com_grappa),[w,h]),abs(com_mostdl)),1),1)*10,[0 1],[1,2]);
title('GRAPPA reconstruction vs MOST-DL (10¡Á)');
figure; imshow3(cat(3,t2_grappa,t2_mostdl),[0 0.2],[1,2]);colormap jet;colorbar;
title('T2 mapping (s) results of GRAPPA reconstruction vs MOST-DL');