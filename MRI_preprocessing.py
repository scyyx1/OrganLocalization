import dicom2nifti
import numpy as np
import SimpleITK as sitk
import glob

document_list = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39] # Specify the list of folders contain MR images
label_directory = "Dataset/MRI/NIFTI_test/label/label-" # Output Label Directory
ite = 20 # Number of MRI images
for i in range(ite):
    directory = "Dataset/MRI/raw_MR/" + str(document_list[i])+"/T2SPIR/" #  Original Raw MRI Directory
    dicom_directory = directory + "DICOM_anon/" # DICOM Directory
    groundtruth_directory = directory + "Ground/" # Ground Truth Directory
    output_file = "Dataset/MRI/NIFTI_test/test-volume-"+str(i) + ".nii" # Output Path
    dicom2nifti.dicom_series_to_nifti(dicom_directory, output_file, reorient_nifti=True) # DICOM to NIFTI


    # Re-process the output nifti file
    raw_image = sitk.ReadImage(output_file)
    data = sitk.GetArrayFromImage(raw_image)
    data = np.ascontiguousarray(np.flip(data, axis=1)) # Change its orientation
    out = sitk.GetImageFromArray(data)
    out.SetSpacing(raw_image.GetSpacing())
    out.SetOrigin(raw_image.GetOrigin())
    sitk.WriteImage(out, output_file)

    # Generate Ground Truth Label NIFTI Images and Set the Spacing and Direction
    direction = out.GetDirection()
    spacing = raw_image.GetSpacing()
    file_names = glob.glob(groundtruth_directory+'*.png')
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    vol = reader.Execute()
    vol.SetSpacing(spacing)
    vol.SetDirection(direction)
    sitk.WriteImage(vol, label_directory + str(i)+".nii")




