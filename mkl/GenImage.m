imname= 'image1.jpg';
X= imread(imname);
A= [X(:,:,1); X(:,:,2); X(:,:,3)]; 
fid = fopen("image1.dat", "w");
fwrite(fid, A, "uint8");
fclose(fid);