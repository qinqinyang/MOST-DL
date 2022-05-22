clc
clear
addpath('tools');
source='./Sub1_ACS_data/';
tar='./coilmaps/';
if exist(tar)==0
    mkdir(tar);
end

file_list=dir([source,'*.mat']);
file_n=length(file_list);
w=128;
h=128;
c=16;

for iii=1:file_n
    file_fn=[source,file_list(iii).name];
    load(file_fn);
    label_data=acs_cal;
    eigThresh_1 = 0.02;
    ksize = [9,9];
    [k,S] = dat2Kernel(label_data,ksize);
    idx = max(find(S >= S(1)*eigThresh_1));
    [M,W] = kernelEig(k(:,:,:,1:idx),[w,h]);
    coil_map=M(:,:,:,end);
    target=[tar,'Sub1_',file_list(iii).name];
    save(target,'coil_map');
    disp(target);
end
