function t2map = T2fit(PDw,T2w,TE)
    t2map=abs(-TE./log(T2w./PDw));
end