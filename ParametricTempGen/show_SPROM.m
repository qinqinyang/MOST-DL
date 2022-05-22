clc;
clear;
addpath('tools');

w=512;
h=512;

T2fn='./Template_SPROM/brain_templates.T2';
M0fn='./Template_SPROM/brain_templates.M0';

% read T2
fid = fopen(T2fn, 'r');
data_in = fread(fid,'float')';
step = length(data_in)/w/h;
T2=reshape(data_in,[w,h,step]);
fclose(fid);

% read M0
fid = fopen(M0fn, 'r');
data_in = fread(fid,'float')';
step = length(data_in)/w/h;
M0=reshape(data_in,[w,h,step]);
fclose(fid);

figure;
imshow3(T2(:,:,1:100),[0 0.2],[10 10]),colormap jet;
figure;
imshow3(M0(:,:,1:100),[0 0.035],[10 10]),colormap gray;