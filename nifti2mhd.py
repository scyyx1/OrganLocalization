import os
import numpy as np
import itk

pname = 'Dataset/MRI/NIFTI_mulGM_test' # modify this path 'pname' to where you store the original NIFTI data
imtype_src = itk.Image[itk.F, 3]
for f in os.listdir(pname):
    if f.endswith('.nii'):
        fname = f.split('.nii')[0]
        print(fname)
        fid = int(fname.split('-')[2]) # modify this index to 1 if it's LiTS training image
        if fname.startswith('test-volume-'): # modify this to 'volume-' if it's LiTS training image
            imtype = itk.Image[itk.SS, 3]
        else:
            imtype = itk.Image[itk.UC, 3]
        print(fname)
        reader = itk.ImageFileReader[imtype_src].New()
        reader.SetFileName('{}/{}'.format(pname, f))
        reader.Update()
        image_src = reader.GetOutput()

        image_arr = itk.GetArrayFromImage(image_src)
        eql = True
        for i in range(3):
            for j in range(3):
                if i==j and image_src.GetDirection().GetVnlMatrix().get(i,j) != 1 or i!=j and image_src.GetDirection().GetVnlMatrix().get(i,j) != 0:
                    eql = False
                    break
            if not eql:
                break
        if not eql:
            image_arr = np.ascontiguousarray(np.flip(image_arr, axis=1))
            if fid >= 68 and fid <= 82: # For LiTS Training image and images from other dataset
            # if fid >= 15 and fid <= 29: # For LiTS Testing image
                image_arr = np.ascontiguousarray(np.flip(image_arr, axis=2))
            print('{}: {} {} {} {} {} {} {} {} {}'.format(
                fname,
                image_src.GetDirection().GetVnlMatrix().get(0,0),
                image_src.GetDirection().GetVnlMatrix().get(0,1),
                image_src.GetDirection().GetVnlMatrix().get(0,2),
                image_src.GetDirection().GetVnlMatrix().get(1,0),
                image_src.GetDirection().GetVnlMatrix().get(1,1),
                image_src.GetDirection().GetVnlMatrix().get(1,2),
                image_src.GetDirection().GetVnlMatrix().get(2,0),
                image_src.GetDirection().GetVnlMatrix().get(2,1),
                image_src.GetDirection().GetVnlMatrix().get(2,2)
                ))
        else:
            print(fname)

        image = itk.GetImageFromArray(image_arr)
        image.SetSpacing(image_src.GetSpacing())
        image.SetOrigin(image_src.GetOrigin())

        cast = itk.CastImageFilter[imtype_src, imtype].New()
        cast.SetInput(image)
        cast.Update()
        image = cast.GetOutput()

        writer = itk.ImageFileWriter[imtype].New()
        writer.SetInput(image)
        writer.SetFileName('{}/{}.mhd'.format(pname, fname))
        writer.Update()