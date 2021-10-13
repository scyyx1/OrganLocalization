import SimpleITK as sitk

# Specify Input Image Path and Output Image Path

# input_directory = "Dataset/LiTS_data/NIFTI_train/volume-"
# output_directory = "Dataset/LiTS_data/NIFTI_gradient_train/volume-"

# input_directory = "Dataset/BTCV_data/raw_NIFTI/test-volume-"
# output_directory = "Dataset/BTCV_data/raw_gradient_NIFTI/test-volume-"

input_directory = "Dataset/LiTS_data/NIFTI_test/test-volume-"
output_directory = "Dataset/LiTS_data/NIFTI_gradient_test/test-volume-"

ite = 70 # Number of images to process (130 for training set of LiTS, 70 for testing set of LiTS, 30 for BTCV)
for i in range(ite):
    itk_img = sitk.ReadImage(input_directory+str(i)+".nii")
    pixel_type = itk_img.GetPixelIDTypeAsString()
    if pixel_type != "32-bit float":
        itk_img = sitk.Cast(itk_img, sitk.sitkFloat32)  # Change pixel type to float to calculate gradient map
    gradient_img = sitk.SobelEdgeDetection(itk_img)
    data = sitk.GetArrayFromImage(gradient_img)
    data[data >= 5000] = 5000 # Threshold to remove some noise
    out = sitk.GetImageFromArray(data)
    out.SetOrigin(gradient_img.GetOrigin())
    out.SetSpacing(gradient_img.GetSpacing())
    out.SetDirection(gradient_img.GetDirection())
    out = sitk.Cast(out, sitk.sitkInt16) # Re-set the pixel type to signed integer 16 to save space
    sitk.WriteImage(out, output_directory + str(i) + ".nii")

