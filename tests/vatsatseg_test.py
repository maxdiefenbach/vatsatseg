import os
import subprocess


def test_cli():

    water = 'nifti/t1_vibe_dixon_tra_lower_W.nii'
    fat = 'nifti/t1_vibe_dixon_tra_lower_F.nii'
    output = 'nifti/vat_sat_seg.nii'

    if os.path.exists(output):
        os.remove(output)

    subprocess.call(['vatsatseg', '-w', water, '-f', fat, '-o', output])

    assert os.path.exists(output)
