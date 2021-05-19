transferImage = gather(extractdata(dlOutput));
transferImage = transferImage + meanVggNet;
transferImage = uint8(transferImage);
transferImage = imresize(transferImage,size(contentImage,[1 2]));
imshow(imtile({contentImage,transferImage,styleImage}, ...
    'GridSize',[1 3],'BackgroundColor','w'));