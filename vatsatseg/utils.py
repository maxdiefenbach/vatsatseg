import numpy as np
import SimpleITK as sitk
from skimage.util import montage
import matplotlib.pyplot as plt


def create_overlay(image, label, opacity=0.3):
    label = sitk.Cast(label, sitk.sitkUInt8)
    return sitk.LabelOverlay(sitk.Cast(sitk.RescaleIntensity(image),
                                       label.GetPixelID()),
                             label, opacity=opacity)


def plot_montage(ndimg, slicenumbers=True, **fig_kwargs):
    m = montage_ndimage(ndimg)
    plt.figure(**fig_kwargs)
    plt.imshow(m)


def montage_ndimage(ndimg):
    if np.ndim(ndimg) == 3:
        m = montage(ndimg, fill=0)
    elif np.ndim(ndimg) == 4:  # + rgb dimension
        m = np.stack((montage(np.squeeze(ndimg[..., 0]), fill=0),
                      montage(np.squeeze(ndimg[..., 1]), fill=0),
                      montage(np.squeeze(ndimg[..., 2]), fill=0)),
                     axis=2)
    return m
