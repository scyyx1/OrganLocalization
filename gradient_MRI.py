import SimpleITK as sitk
# Specify Input Image Path and Output Image Path

input_directory = "Dataset/MRI/NIFTI_test/test-volume-"
output_directory = "Dataset/MRI/NIFTI_gradient_test/test-volume-"

ite = 20 # Number of images to process
for i in range(ite):
    itk_img = sitk.ReadImage(input_directory+str(i)+".nii")
    pixel_type = itk_img.GetPixelIDTypeAsString()
    if pixel_type != "32-bit float":
        itk_img = sitk.Cast(itk_img, sitk.sitkFloat32) # Change pixel type to float to calculate gradient map
    gradient_img = sitk.SobelEdgeDetection(itk_img)
    data = sitk.GetArrayFromImage(gradient_img)
    data[data >= 5000] = 5000 # Threshold to remove some noise
    out = sitk.GetImageFromArray(data)
    out.SetOrigin(gradient_img.GetOrigin())
    out.SetSpacing(gradient_img.GetSpacing())
    out.SetDirection(gradient_img.GetDirection())
    out = sitk.Cast(out, sitk.sitkInt16)
    sitk.WriteImage(out, output_directory + str(i) + ".nii") # Re-set the pixel type to signed integer 16 to save space
    print(i)