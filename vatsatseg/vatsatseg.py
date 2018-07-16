import numpy as np
import SimpleITK as sitk
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from skimage.morphology import label, convex_hull_image
from dask import delayed, compute
from dask.distributed import Client
import click
import subprocess
import os
import configparser


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-w', '--water', type=click.Path(exists=True),
              help='Water MRI image file.')
@click.option('-f', '--fat', type=click.Path(exists=True),
              help='Fat MRI image file.')
@click.option('-o', '--output', type=click.Path(exists=False),
              help='Output label map file name.')
@click.option('--peel', default=10, show_default=True,
              help='Number of voxels of the subcutaneous fat ring thickness.')
@click.option('-s', '--show', is_flag=True,
              help='Open ITK-SNAP to manually correct the labelmap.')
def vat_sat_seg(water, fat, output, peel, show):
    """
    segment Visceral and Subcutaneous Adipose Tissue (VAT, SAT)
    in water-fat MRI images
    """
    if water is None or fat is None:
        with click.Context(vat_sat_seg, info_name='vatsatseg') as ctx:
            click.echo(ctx.get_help())
            return

    config = configparser.ConfigParser()
    wd = os.path.dirname(__file__)
    config.read(os.path.join(wd, 'config.ini'))

    print('Read input. ', end='')
    Water = sitk.ReadImage(water)
    Fat = sitk.ReadImage(fat)

    w_img = sitk.GetArrayFromImage(Water)
    f_img = sitk.GetArrayFromImage(Fat)
    print('Done.')

    print('Start segmentation. ', end='')
    labelmap = vat_sat_seg_parallel2d(w_img, f_img, peel)
    print('Done.')

    label_img = sitk.GetImageFromArray(labelmap.astype(float))
    label_img.CopyInformation(Water)
        
    sitk.WriteImage(label_img, output)
    print('Wrote labelmap "{}".'.format(output))

    if show:
        print('Open viewer.')
        viewer = config['viewer']
        cmd = [viewer['cmd']] + \
              viewer['opts'].format(water, fat, output,
                                    os.path.join(wd, viewer['labeldescfile'])) \
                            .split(' ')
        print('Open viewer "{}".'.format(viewer['cmd']))
        subprocess.call(cmd)


def vat_sat_seg_parallel2d(w_img, f_img, peel=10):
    b_mask, w_mask, f_mask = kmeans(w_img, f_img)

    results = []
    with Client() as client:
        for iz in range(b_mask.shape[0]):
            results.append(delayed(get_labelmap_2d)(b_mask[iz, :, :],
                                                    w_mask[iz, :, :],
                                                    f_mask[iz, :, :],
                                                    peel))
        results = compute(*results)

    return np.array(results)


def get_labelmap_2d(b_mask, w_mask, f_mask, peel):
    vat_mask, sat_mask, torso_mask = \
                        differentiate_vat_sat(b_mask, w_mask, f_mask, peel)

    labelmap = np.zeros_like(b_mask, dtype=int)
    labelmap[w_mask & torso_mask] = 1
    labelmap[vat_mask] = 2
    labelmap[sat_mask] = 3

    return labelmap


def vat_sat_seg_slice(w_img, f_img, peel=10):
    
    results = []
    with Client() as client:
        for iz in range(w_img.shape[0]):
            results.append(delayed(vat_sat_seg_2d)(w_img[iz, :, :],
                                                   f_img[iz, :, :],
                                                   peel))
        results = compute(*results)

    return np.array(results)


def vat_sat_seg_2d(w_img, f_img, peel=10):
    b_mask, w_mask, f_mask = kmeans(w_img, f_img)
    vat_mask, sat_mask, torso_mask = \
                        differentiate_vat_sat(b_mask, w_mask, f_mask, peel)

    labelmap = np.zeros_like(b_mask, dtype=int)
    labelmap[w_mask & torso_mask] = 1
    labelmap[vat_mask] = 2
    labelmap[sat_mask] = 3

    return labelmap


def kmeans(w_img, f_img):
    # number of observations = number of voxels
    # two features: water, fat
    data = np.array([w_img.ravel(), f_img.ravel()]).T
    
    model = MiniBatchKMeans(n_clusters=3) # background, water, fat
    model.fit(data)
    
    sz = w_img.shape
    labelmap = model.labels_.reshape(sz)
    
    # sort clusters (labels) by water feature
    labels = np.argsort(model.cluster_centers_[:, 0])
    background_mask = labelmap == labels[0]
    fat_mask = labelmap == labels[1]
    water_mask = labelmap == labels[2]
    
    return background_mask, water_mask, fat_mask


def get_labelmap_2d(b_mask, w_mask, f_mask, peel):
    vat_mask, sat_mask, torso_mask = \
                        differentiate_vat_sat(b_mask, w_mask, f_mask, peel)

    labelmap = np.zeros_like(b_mask, dtype=int)
    labelmap[w_mask & torso_mask] = 1
    labelmap[vat_mask] = 2
    labelmap[sat_mask] = 3

    return labelmap


def differentiate_vat_sat(background_mask, water_mask, fat_mask,
                          peel=10):
    torso_mask = extract_torso_mask(background_mask)
    noskin_mask = binary_erosion(torso_mask, iterations=peel)
    inner_torso_mask = convex_hull_image(noskin_mask & water_mask)
    
    vat_mask = inner_torso_mask & fat_mask
    sat_mask = torso_mask & fat_mask & ~vat_mask
    
    return vat_mask, sat_mask, torso_mask
    
    
def extract_torso_mask(background_mask):
    bw_filled = binary_fill_holes(~background_mask)
    
    labels = label(bw_filled)
    labelCount = np.bincount(labels.ravel())
    
    # assume most pixels are background
    torso_mask = labels == np.argsort(labelCount)[-2]

    return convex_hull_image(torso_mask)


if __name__ == '__main__':

    vat_sat_seg()
