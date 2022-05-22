clc;
clear;
addpath('tools');

map_path = './Paramap/';
if ~exist(map_path,'dir')==1
    mkdir(map_path);
end

PD_path = './IXI/IXI-PD/';
T2_path = './IXI/IXI-T2/';
PDlist = dir([PD_path,'*.nii.gz']);
T2list = dir([T2_path,'*.nii.gz']);
w=256;h=256;

for loopdir = 1:length(PDlist)
    tar_t2 = [T2_path,T2list(loopdir).name];
    tar_pd = [PD_path,PDlist(loopdir).name];
    
    tar_map = [map_path,'Paramap_',num2str(loopdir),'.mat'];
    
    % load IXI raw data
    T2=load_untouch_nii(tar_t2);
    T2=double(T2.img);
    PD=load_untouch_nii(tar_pd);
    PD=double(PD.img);
    
    slicen=size(PD,3);
    
    slice_list = [35:2:slicen-25];
    mat_pd=zeros(w,h,length(slice_list));
    mat_t2=zeros(w,h,length(slice_list));
    
    for i=1:length(slice_list)
        PDt=PD(:,:,slice_list(i));
        T2t=T2(:,:,slice_list(i));
        PDtt=abs(PDt/max(PDt(:)));
        
        % mask
        level = 0.1;
        mask=im2bw(PDtt,level);
        
        TE=(60*rand()+30)*1e-3; % randomize TE for various T2 distribution
        T2t = T2fit(PDt,T2t,TE);
        T2t(T2t>0.65)=0.65;
        T2t(isnan(T2t))=0;T2t(T2t==inf)=0;
        
        mat_t2(:,:,i)=T2t.*mask;
        mat_pd(:,:,i)=PDtt.*mask;
    end
    save(tar_map,'mat_t2','mat_pd');
    disp(['Finish file:',tar_pd]);
end