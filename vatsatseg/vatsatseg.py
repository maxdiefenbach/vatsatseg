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
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
def cli(water, fat, output, peel, show):
    """
    segment Visceral and Subcutaneous Adipose Tissue (VAT, SAT)
    in water-fat MRI images
    """
    if water is None or fat is None:
        with click.Context(cli, info_name='vatsatseg') as ctx:
            click.echo(ctx.get_help())
            return

    config = configparser.ConfigParser()
    wd = os.path.dirname(__file__)
    config.read(os.path.join(wd, 'config.ini'))

    vatsatseg(water, fat, output, peel)

    if show:
        logger.info('Open viewer.')
        viewer = config['viewer']
        cmd = [viewer['cmd']] + \
              viewer['opts'].format(water, fat, output,
                                    os.path.join(wd, viewer['labeldescfile'])) \
                            .split(' ')
        logger.info('Open viewer "{}".'.format(viewer['cmd']))
        subprocess.call(cmd)


def vatsatseg(water, fat, output, peel=None):
    """ perform vatsatseg by reading in filenames for water, fat and output
    file extensions can be anything understood by SimpleITK

    :param water: str, filepath to water image file
    :param fat: str, filepath to fat image file
    :param output: str, filepath to output labelmap
    :param peel: int, default=10, dilation radius on the water mask big enough
                                  to crop watery skin
    :returns:
    :rtype:

    """

    logger.info('Read input. ')
    Water = sitk.ReadImage(water)
    Fat = sitk.ReadImage(fat)

    w_img = sitk.GetArrayFromImage(Water)
    f_img = sitk.GetArrayFromImage(Fat)
    logger.info('Done.')

    logger.info('Start segmentation.')
    labelmap = vatsatseg_parallel2d(w_img, f_img, peel)
    logger.info('Done.')

    label_img = sitk.GetImageFromArray(labelmap.astype(float))
    label_img.CopyInformation(Water)

    sitk.WriteImage(label_img, output)
    logger.info('Wrote labelmap "{}".'.format(output))


def vatsatseg_parallel2d(w_img, f_img, peel=10):
    """runs vatsatseg on 3d arrays slice by slice in parrallel

    :param w_img: 3d numpy.ndarray
    :param f_img: 3d numpy.ndarray
    :param peel: int, default=10, dilation radius on the water mask big enough
                                  to crop watery skin
    :returns: labelmap with 4 components: background 0, water 1, vat 2, sat 3
    :rtype: 3d numpy.ndarray

    """
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
    """after kmeans clustering of background water and fat in one slice
    run logic to differentiate vat and sat and create labelmap

    :param b_mask: 2d numpy.ndarray, binary background mask
    :param w_mask: 2d numpy.ndarray, binary water mask
    :param f_mask: 2d numpy.ndarray, binary fat mask
    :param peel: int, default=10, dilation radius on the water mask big enough
                                  to crop watery skin
    :returns: labelmap with 4 components: background 0, water 1, vat 2, sat 3
    :rtype: 2d numpy.ndarray

    """
    vat_mask, sat_mask, torso_mask = \
                        differentiate_vat_sat(b_mask, w_mask, f_mask, peel)

    labelmap = np.zeros_like(b_mask, dtype=int)
    labelmap[w_mask & torso_mask] = 1
    labelmap[vat_mask] = 2
    labelmap[sat_mask] = 3

    return labelmap


# def vatsatseg_slice(w_img, f_img, peel=10):

#     results = []
#     with Client() as client:
#         for iz in range(w_img.shape[0]):
#             results.append(delayed(vatsatseg_2d)(w_img[iz, :, :],
#                                                    f_img[iz, :, :],
#                                                    peel))
#         results = compute(*results)

#     return np.array(results)


# def vatsatseg_2d(w_img, f_img, peel=10):
#     b_mask, w_mask, f_mask = kmeans(w_img, f_img)
#     vat_mask, sat_mask, torso_mask = \
#                         differentiate_vat_sat(b_mask, w_mask, f_mask, peel)

#     labelmap = np.zeros_like(b_mask, dtype=int)
#     labelmap[w_mask & torso_mask] = 1
#     labelmap[vat_mask] = 2
#     labelmap[sat_mask] = 3

#     return labelmap


def kmeans(w_img, f_img):
    """perform k-means clustering with k=3 for water, fat and background
    given input arrays for water and fat

    :param w_img: numpy.ndarray for water images
    :param f_img: numpy.ndarray for fat images
    :returns: background-, water- and fat mask
    :rtype: tuple of 3 numpy.ndarrays

    """
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


def differentiate_vat_sat(background_mask, water_mask, fat_mask, peel=0):
    """masking logic to differentiate vat from sat

    :param background_mask: numpy.ndarray
    :param water_mask: numpy.ndarray
    :param fat_mask: numpy.ndarray
    :param peel: int, default=10, dilation radius on the water mask big enough
                                  to crop watery skin
    :returns: vat_mask, sat_mask, torso_mask
    :rtype: tuple holding three numpy.ndarrays

    """
    torso_mask = extract_torso_mask(background_mask)
    peeled_water_mask = binary_erosion(water_mask, iterations=peel) \
        if peel > 0 else water_mask
    inner_torso_mask = extract_inner_torso_mask(peeled_water_mask)

    vat_mask = inner_torso_mask & fat_mask
    sat_mask = torso_mask & fat_mask & ~vat_mask

    return vat_mask, sat_mask, torso_mask


def extract_torso_mask(background_mask):
    """create mask for the torso
    (arms might be included if there is a path bridging them to the torso)

    :param background_mask: numpy.ndarray
    :returns: torso_mask
    :rtype: numpy.ndarray

    """
    bw_filled = binary_fill_holes(~background_mask)

    labels = label(bw_filled)
    labelCount = np.bincount(labels.ravel())

    # assume most pixels are background
    torso_mask = labels == np.argsort(labelCount)[-2]

    return convex_hull_image(torso_mask)


def extract_inner_torso_mask(w_mask):
    """given the water mask, try to extract a smaller "inner torso" mask
    for only the inner region starting from the subcutaneous fat

    :param w_mask: numpy.ndarray, water mask
    :returns: inner torso mask
    :rtype: numpy.ndarray

    """
    bw_filled = binary_fill_holes(w_mask)
    labels = label(bw_filled)
    labelCount = np.bincount(labels.ravel())
    inner_torso_mask = labels == np.argsort(labelCount)[-2]
    return convex_hull_image(inner_torso_mask)


if __name__ == '__main__':

    cli()
