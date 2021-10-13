import itk

# Specify Image Path, Result Path, Ground Truth Path, Output Path
img_path = "Dataset/MRI/Testing/img/"
result_path = "Dataset/MRI/Testing/img/results/"
ground_truth = "Dataset/MRI/Testing/img/ground_truth/"
output_path = "Dataset/MRI_crop_image/"
# img_path = "Dataset/BTCV_data/Testing/img/"
# result_path = "Dataset/BTCV_data/Testing/img/results/"
# ground_truth = "Dataset/BTCV_data/Testing/img/ground_truth/"
# output_path = "Dataset/BTCV_crop_image/"
# img_path = "Dataset/LiTS_data/Testing/img/"
# result_path = "Dataset/LiTS_data/Testing/img/results/"
# ground_truth = "Dataset/LiTS_data/Testing/img/ground_truth/"
# output_path = "Dataset/LiTS_crop_image/"

# Crop Selected Image with organ order (organ_order = which line of organs user want to crop)
def crop_selected_img(image_path, result_path, organ_order, output_path, is_prediction):

    with open(result_path, "r") as file:
        ite = organ_order
        organs = ""
        while(ite):
            organs = file.readline()
            ite -= 1
        ele_row = organs.split(" ")
        size_x = 0
        size_y = 0
        size_z = 0
        if(is_prediction):
            size_x = int(float(ele_row[4])) - int(float(ele_row[1]))
            size_y = int(float(ele_row[5])) - int(float(ele_row[2]))
            size_z = int(float(ele_row[6])) - int(float(ele_row[3]))
        else:
            size_x = int(ele_row[3]) - int(ele_row[2])
            size_y = int(ele_row[5]) - int(ele_row[4])
            size_z = int(ele_row[7].strip('\n')) - int(ele_row[6])
        reader = itk.ImageFileReader.New(FileName=image_path)
        reader.Update()
        image = reader.GetOutput()
        cropper = itk.ExtractImageFilter.New(Input=image)
        cropper.SetDirectionCollapseToIdentity()
        extraction_region = cropper.GetExtractionRegion()
        size = extraction_region.GetSize()
        size[0] = int(size_x)
        size[1] = int(size_y)
        size[2] = int(size_z)
        index = extraction_region.GetIndex()
        if (is_prediction):
            index[0] = int(float(ele_row[1]))
            index[1] = int(float(ele_row[2]))
            index[2] = int(float(ele_row[3])) + 1
        else:
            index[0] = int(float(ele_row[2]))
            index[1] = int(float(ele_row[4]))
            index[2] = int(float(ele_row[6])) + 1
        print(index[0], index[1], index[2])
        extraction_region.SetSize(size)
        extraction_region.SetIndex(index)
        cropper.SetExtractionRegion(extraction_region)
        itk.ImageFileWriter.New(Input=cropper, FileName=output_path).Update()

# Crop Predicted Results
crop_selected_img(image_path=img_path+"test-volume-0.mhd",
                  result_path=result_path+"test-volume-0.mhd.pred.txt",
                  organ_order=1,
                  output_path=output_path+"prediction_liver.nii",
                  is_prediction=True)
# Crop Ground Truth Results
crop_selected_img(image_path=img_path+"test-volume-0.mhd",
                  result_path=ground_truth+"test-segmentation-0.txt",
                  organ_order=1,
                  output_path=output_path+"ground_truth_liver.nii",
                  is_prediction=False)

