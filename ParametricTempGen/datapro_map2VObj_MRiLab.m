clc;
clear;

map_path = './Paramap/';
temp_path = './Template_MRiLab/';
if ~exist(temp_path,'dir')==1
    mkdir(temp_path);
end

expand=512;

maplist=dir([map_path,'*.mat']);
mapn=length(maplist);
slicen=36;
order=1;

% VObj
VObj.Gyro=2.675380303797068e+08;
VObj.ChemShift=0;
VObj.XDim=expand;
VObj.YDim=expand;
VObj.ZDim=1;
VObj.XDimRes=4.4e-04;
VObj.YDimRes=4.4e-04;
VObj.ZDimRes=1.0e-04;
VObj.Type='Water';
VObj.TypeNum=1;
VObj.ECon=[];
VObj.MassDen=[];

for loopdir = 1:mapn
    mapfn=[map_path,maplist(loopdir).name];
    load(mapfn);
    
    for slicei=1:slicen
        tarname=[temp_path,num2str(order),'.mat'];
        ro=randperm(4);
        fl=randperm(2);
        t2=flip(rot90(abs(mat_t2(:,:,slicei)),ro(1)),fl(1));
        pd=flip(rot90(abs(mat_pd(:,:,slicei)),ro(1)),fl(1));
        t2=imresize(t2,[expand,expand]);
        pd=imresize(pd,[expand,expand]);
        t1=zeros(expand,expand);t1(pd>0)=2;
        
        VObj.Rho=abs(pd);
        VObj.T2=abs(t2).*(pd>0);
        VObj.T2Star=VObj.T2;
        VObj.T1=t1;
        save(tarname,'VObj');
        disp(tarname);
        order=order+1;
    end
end