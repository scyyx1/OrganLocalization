clear all;
close all;

% [LiTS_param, BTCV_param, MRI_param]
level_sets = [4, 3, 3];
window_sets = [3, 3, 13];
threshold_sets = [7, 7, 7];
ite = 20; % 130 for LiTS train data, 70 for LiTS test data, 30 for BTCV test data, 20 for MRI test data
for cur = 1:ite
    img = single(niftiread("Dataset/MRI/NIFTI_mulGM_test/test-volume-"+(cur-1)+".nii"));
    info = niftiinfo("Dataset/MRI/NIFTI_mulGM_test/test-volume-"+(cur-1)+".nii");
    if info.Datatype ~= "int16"
       info.Datatype = "int16";
    end

    slice_num = size(img, 3);
    for slice = 1: slice_num
        I = img(:, :, slice);

        nLevel=level_sets(3);  % level setting depends on img size
        imgScale=1./(2.^[0:nLevel-1]);
        hW=window_sets(3);       % window size for normalisation
        mulGM=zeros(size(I));

        for level=1:nLevel
            tmpI=imresize(I,imgScale(level));
            [GtmpI,~]=imgradient(tmpI);
            stdI=std(GtmpI(:));
            norm_GM=zeros(size(GtmpI));
            for r=hW+1:size(GtmpI,1)-hW
                for c=hW+1:size(GtmpI,2)-hW
                    tmpI=GtmpI(r-hW:r+hW,c-hW:c+hW);
                    if std(tmpI(:))<stdI/threshold_sets(3)     % threshold to remove homogenous regions 
                        norm_GM(r,c)=0;
                    else
                        tmpI=(tmpI-min(tmpI(:)))/(max(tmpI(:))-min(tmpI(:)));
                        norm_GM(r,c)=tmpI(hW+1,hW+1);
                    end
                end
            end
            mulGM=mulGM+imresize(norm_GM,[size(I,1), size(I,2)]);
        end

        mulGM=(mulGM-min(mulGM(:)))/(max(mulGM(:))-min(mulGM(:)));
        img(:, :, slice) = mulGM;
    end
    img = int16(img*255);
    niftiwrite(img, "Dataset/MRI/NIFTI_mulGM_test/test-volume-"+(cur-1)+".nii", info);
    disp("Current Img: " + cur + " Remaining Img: "+ (ite - cur));
end





