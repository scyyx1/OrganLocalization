import nibabel as nib
import numpy as np

# Get the Organs Name Using Label
organ_dict = {"liver": 6, "kidney-r": 2, "kidney-l": 3, "bladder": 4, "spleen": 1, "pancreas": 11}

directory = "Dataset/BTCV_data/Testing/label/" # Annotated Image Directory

ite = 30 # Number of images to extract labels
for i in range(ite):
        img = nib.load(directory+"label-"+str(i)+".nii")
        data = img.get_fdata()
        with open(directory+"test-segmentation-"+str(i)+".txt", "w+") as f:
            for organ, label in organ_dict.items():
                a = np.where(data == label) # Find all specific pixels location
                if a[0].size == 0: # if cannot find any pixels, we skip that organ
                    continue
                x0 = min(a[0])
                x1 = max(a[0])
                y0 = min(a[1])
                y1 = max(a[1])
                z0 = min(a[2])
                z1 = max(a[2])
                # Write all locations to file
                f.writelines(organ+" "+str(label)+" "+str(x0)+" "+str(x1)+" "+str(y0)+" "+str(y1)+" "+str(z0)+" "+str(z1)+"\n")
