import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from skimage.morphology import label, convex_hull_image
from dask import delayed, compute
from dask.distributed import Client
import click
import subprocess
import configparser


config = configparser.ConfigParser()
config.read('config.ini')


@click.command()
@click.option('-w', '--water')
@click.option('-f', '--fat')
@click.option('-o', '--output')
@click.option('--peel', default=10)
@click.option('-s', '--show', is_flag=True)
def vat_sat_seg(water, fat, output, peel, show):
    Water = sitk.ReadImage(water)
    Fat = sitk.ReadImage(fat)

    w_img = sitk.GetArrayFromImage(Water)
    f_img = sitk.GetArrayFromImage(Fat)

    labelmap = vat_sat_seg_slice(w_img, f_img, peel)
    # labelmap = vat_sat_seg_global(w_img, f_img, peel)

    label_img = sitk.GetImageFromArray(labelmap.astype(float))
    label_img.CopyInformation(Water)
        
    sitk.WriteImage(label_img, output)
    print('Wrote "{}".'.format(output))

    if show:
        viewer = config['viewer']
        cmd = [viewer['cmd']] + \
              viewer['opts'].format(water, fat, output,
                                    viewer['labeldescfile']) \
                            .split(' ')
        subprocess.Popen(cmd)


def vat_sat_seg_global(w_img, f_img, peel=10):
    b_mask, w_mask, f_mask = kmeans(w_img, f_img)

    # with Client() as client:
    client = Client()

    results = []
    for iz in range(b_mask.shape[0]):
        results.append(delayed(get_labelmap_2d)(b_mask[iz, :, :],
                                                w_mask[iz, :, :],
                                                f_mask[iz, :, :],
                                                peel))

    results = compute(*results)

    client.close()

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
    client = Client()
    
    # with Client() as client:
    results = []
    for iz in range(w_img.shape[0]):
        results.append(delayed(vat_sat_seg_2d)(w_img[iz, :, :],
                                               f_img[iz, :, :],
                                               peel))

    results = compute(*results)

    client.close()

    labelmap = np.array(results)

    return labelmap


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
