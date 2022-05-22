n=1;
m=4;
figure(31)
subplot(n,m,1)
imagesc(VObj.T1),colormap jet;colorbar;axis image;
set(gca,'xtick',[],'xticklabel',[]);
set(gca,'ytick',[],'yticklabel',[]);
title('T1');

subplot(n,m,2)
imagesc(VObj.T2,[0 0.2]),colormap jet;colorbar;axis image;
set(gca,'xtick',[],'xticklabel',[]);
set(gca,'ytick',[],'yticklabel',[]);
title('T2');

subplot(n,m,3)
imagesc(VObj.T2Star,[0 0.2]),colormap jet;colorbar;axis image;
set(gca,'xtick',[],'xticklabel',[]);
set(gca,'ytick',[],'yticklabel',[]);
title('T2*');

subplot(n,m,4)
imagesc(VObj.Rho,[0 1]),colormap jet;colorbar;axis image;
set(gca,'xtick',[],'xticklabel',[]);
set(gca,'ytick',[],'yticklabel',[]);
title('M0');
