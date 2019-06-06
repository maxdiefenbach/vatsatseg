

# Description

`vatsatseg` is a python implementation of the matlab segmentation tool "SAT\_VAT\_segmentation" used in

*Shen, J., Baum, T., Cordes, C., Ott, B., Skurk, T., Kooijman, H., Rummeny, E. J., …, Automatic segmentation of abdominal organs and adipose tissue compartments in water-fat mri: application to weight-loss in obesity, European Journal of Radiology, 85(9), 1613–1621 (2016).*
<http://dx.doi.org/10.1016/j.ejrad.2016.06.006>


# Installation

You can install `vatsatseg` by typing the following commands in your terminal:

    git clone https://github.com/maxdiefenbach/vatsatseg # clone project
    
    cd vatsatseg                    # change directory
    
    pip install .                # install package

This automatically installs the commandline tool "$ vatsatseg".


# Dependency

To manually correct the segmentation result `vatsatseg` relies on [ITK-SNAP](http://www.itksnap.org).
By invoking the "&#x2013;show" option, the water and fat images together with the segmentation overlay will be opened. For this you need to make sure the "viewer command" in the configuration file "*vatsatseg/config.ini*" points to the correct installation path of ITK-SNAP.


# Tutorial


## see help

    vatsatseg --help

    Usage: vatsatseg [OPTIONS]
    
      segment Visceral and Subcutaneous Adipose Tissue (VAT, SAT) in water-fat
      MRI images
    
    Options:
      -w, --water PATH   Water MRI image file.
      -f, --fat PATH     Fat MRI image file.
      -o, --output PATH  Output label map file name.
      -s, --show         Open ITK-SNAP to manually correct the labelmap.
      -h, --help         Show this message and exit.


## example

    vatsatseg \
           -w t1_vibe_dixon_tra_lower_W.nii \
           -f t1_vibe_dixon_tra_lower_F.nii \
           -o vat_sat_seg.nii \
           -s


# Contact

`vatsatseg` is in an early development state. We are happy about feedback and/or contributions on [Github](https://github.com/maxdiefenbach/vatsatseg).


## Report bugs / Request Features

If you can report a bug or would like to request a feature, please create an issue [here](https://github.com/maxdiefenbach/vatsatseg/issues).


## Contribute

If you can contribute we are happy about pull or merge requests on [Github](https://github.com/maxdiefenbach/vatsatseg).


# License

vatsatseg, segment visceral and subcutaneous adipose tissue
Copyright (C) 2018-2019 Maximilian N. Diefenbach

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Acknowlegment

`vatsatseg` is developed in the [Body Magnetic Resonance Research Group](http://www.bmrrgroup.de) at the [Klinikum rechts der Isar](http://www.mri.tum.de/) and the [Technical University Munich](http://www.tum.de/).

