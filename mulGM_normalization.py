import SimpleITK as sitk

# Specify Input Image Path, Info Path and Output Image Path

# input_directory = "Dataset/LiTS_data/NIFTI_mulGM_train/volume-"
# info_directory = "Dataset/LiTS_data/NIFTI_train/volume-"
# output_directory = "Dataset/LiTS_data/NIFTI_mulGM_train/volume-"

# input_directory = "Dataset/LiTS_data/NIFTI_mulGM_test/test-volume-"
# info_directory = "Dataset/LiTS_data/NIFTI_test/test-volume-"
# output_directory = "Dataset/LiTS_data/NIFTI_mulGM_test/test-volume-"

# input_directory = "Dataset/BTCV_data/NIFTI_mulGM_test/test-volume-"
# info_directory = "Dataset/BTCV_data/NIFTI_test/test-volume-"
# output_directory = "Dataset/BTCV_data/NIFTI_mulGM_test/test-volume-"

input_directory = "Dataset/MRI/NIFTI_mulGM_test/test-volume-"
info_directory = "Dataset/MRI/NIFTI_test/test-volume-"
output_directory = "Dataset/MRI/NIFTI_mulGM_test/test-volume-"
ite = 20 # Number of images to process (70 for LiTS, 30 for BTCV and 20 for CHAOS)
for i in range(ite):
    itk_img = sitk.ReadImage(input_directory + str(i) + ".nii")
    info_img = sitk.ReadImage(info_directory + str(i) + ".nii")
    data = sitk.GetArrayFromImage(itk_img)
    # low_threshold = data[0, 0, 0] # Set the low threshold
    high_threshold = 150 # Set the high threshold
    # data[data <= low_threshold] = 0
    data[data >= high_threshold] = 255 # Remove the noise
    out = sitk.GetImageFromArray(data)
    out.SetOrigin(info_img.GetOrigin())
    out.SetSpacing(info_img.GetSpacing())
    out.SetDirection(info_img.GetDirection())
    sitk.WriteImage(out, output_directory + str(i) + ".nii")

