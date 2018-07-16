#!/bin/bash

# vatsatseg -w nifti/t1_vibe_dixon_tra_lower_W.nii -f nifti/t1_vibe_dixon_tra_lower_F.nii -o nifti/vat_sat_seg.nii

python /Users/maxdiefenbach/programs/BMRR/Postprocessing/SAT_VAT_segmentation/vatsatseg/vatsatseg/vatsatseg.py \
       -w /Users/maxdiefenbach/programs/BMRR/Postprocessing/SAT_VAT_segmentation/vatsatseg/tests/nifti/t1_vibe_dixon_tra_lower_W.nii \
       -f /Users/maxdiefenbach/programs/BMRR/Postprocessing/SAT_VAT_segmentation/vatsatseg/tests/nifti/t1_vibe_dixon_tra_lower_F.nii \
       -o /Users/maxdiefenbach/programs/BMRR/Postprocessing/SAT_VAT_segmentation/vatsatseg/tests/nifti/vat_sat_seg.nii


vatsatseg \
    -w /Users/maxdiefenbach/programs/BMRR/Postprocessing/SAT_VAT_segmentation/vatsatseg/tests/nifti/t1_vibe_dixon_tra_lower_W.nii \
    -f /Users/maxdiefenbach/programs/BMRR/Postprocessing/SAT_VAT_segmentation/vatsatseg/tests/nifti/t1_vibe_dixon_tra_lower_F.nii \
    -o /Users/maxdiefenbach/programs/BMRR/Postprocessing/SAT_VAT_segmentation/vatsatseg/tests/nifti/vat_sat_seg.nii
