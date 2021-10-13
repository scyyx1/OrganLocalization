import numpy as np
import SimpleITK as sitk
label_directory = "Dataset/BTCV_data/Testing/label/label-" # Label Directory
ite = 30 # Number of Images
for i in range(ite):

    # Re-orient Label Image
    raw_img_label = sitk.ReadImage(label_directory + str(i) + ".nii")
    data = sitk.GetArrayFromImage(raw_img_label)
    data = np.ascontiguousarray(np.flip(data, axis=1)) # Flip Label Image
    out = sitk.GetImageFromArray(data)
    out.SetSpacing(raw_img_label.GetSpacing())
    out.SetOrigin(raw_img_label.GetOrigin())
    sitk.WriteImage(out, label_directory + str(i) + ".nii")


