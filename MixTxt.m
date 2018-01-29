%clc
clear
path='/home/zc/project/travel_mod-build/label/commit/';

fid = fopen('train0129_d.txt','a');
%for i = 81:110
for i= 1:1200
    %label:1 asphalt, 2 grass, 3 sand
    name=fullfile(path,[num2str(i,'%04d'),'.txt']);
    fid_i = fopen(name,'r');
    Data_i = fread(fid_i);
    
    %write the data into train.txt
    fwrite(fid,Data_i);
    %newline
    %fprintf(fid,'\n');
    fclose(fid_i);
    i
end
fclose(fid);
i
