import os
import subprocess
import numpy as np
import SimpleITK as sitk
import vatsatseg as vts


def test_cli():
    water = 'nifti/t1_vibe_dixon_tra_lower_W.nii'
    fat = 'nifti/t1_vibe_dixon_tra_lower_F.nii'
    output = 'nifti/vatsatseg.nii'

    if os.path.exists(output):
        os.remove(output)

    subprocess.call(['vatsatseg', '-w', water, '-f', fat, '-o', output])

    assert os.path.exists(output)


def test_vatsatseg_parallel2d():
    water = 'nifti/t1_vibe_dixon_tra_lower_W.nii'
    fat = 'nifti/t1_vibe_dixon_tra_lower_F.nii'

    Water = sitk.ReadImage(water)
    Fat = sitk.ReadImage(fat)

    w_img = sitk.GetArrayFromImage(Water)
    f_img = sitk.GetArrayFromImage(Fat)

    labelmap = vts.vat_sat_seg_parallel2d(w_img, f_img, 10)

    assert isinstance(labelmap, np.ndarray)
    assert w_img.shape == labelmap.shape


def test_install():

    subprocess.call(['cd', '..', '&&', 'pip', 'install', '.'])


def test_cli_dicoms():
    water = '20170101_121431_0901_unknown_W.dcm'
    fat = '20170101_121431_0901_unknown_F.dcm'
    output = 'vatsatseg_dicom.nii.gz'

    if os.path.exists(output):
        os.remove(output)

    subprocess.call(['vatsatseg', '-w', water, '-f', fat, '-o', output])
