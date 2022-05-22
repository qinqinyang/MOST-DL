clc;
clear;

map_path = './Paramap/';
temp_path ='./Template_SPROM/';
if ~exist(temp_path,'dir')==1
    mkdir(temp_path);
end

tarfnt2=[temp_path,'brain_templates.T2'];
tarfnm0=[temp_path,'brain_templates.M0'];

expand=512;

maplist=dir([map_path,'*.mat']);
mapn=length(maplist);
slicen=36;

T2=zeros(expand,expand,slicen*mapn);
M0=zeros(expand,expand,slicen*mapn);

order=1;

for loopdir = 1:mapn
    mapfn=[map_path,maplist(loopdir).name];
    load(mapfn);
    
    for slicei=1:slicen
        ro=randperm(4);
        fl=randperm(2);
        t2=flip(rot90(abs(mat_t2(:,:,slicei)),ro(1)),fl(1));
        pd=flip(rot90(abs(mat_pd(:,:,slicei)),ro(1)),fl(1));
        t2=imresize(t2,[expand,expand]);
        pd=imresize(pd,[expand,expand]);
        t2=abs(t2);
        pd=abs(pd);
        t2=t2.*(pd>0);
        
        T2(:,:,order)=t2;
        M0(:,:,order)=pd;
        order=order+1;
    end
    disp(['Finished N0.',num2str(loopdir)]);
end

T2=single(T2);
M0=single(M0)*0.035;

% save T2 templates
[fid,msg]=fopen(tarfnt2,'wb');
fwrite(fid,T2,'float');
fclose(fid);
disp(['T2 template is saved in ',tarfnt2]);

% save M0 templates
[fid,msg]=fopen(tarfnm0,'wb');
fwrite(fid,M0,'float');
fclose(fid);
disp(['M0 template is saved in ',tarfnm0]);