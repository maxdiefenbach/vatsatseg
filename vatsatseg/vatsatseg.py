import subprocess
import os
import re
import configparser
import logging
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from skimage.morphology import label, convex_hull_image
from skimage.measure import regionprops
import click


# init logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


# command line options
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-w', '--water', type=click.Path(exists=True),
              help='Water MRI image file.')
@click.option('-f', '--fat', type=click.Path(exists=True),
              help='Fat MRI image file.')
@click.option('-o', '--output', type=click.Path(),
              help='Output label map file name.')
@click.option('-s', '--show', is_flag=True,
              help='Open ITK-SNAP to manually correct the labelmap.')
def cli(water, fat, output, show):
    """
    segment Visceral and Subcutaneous Adipose Tissue (VAT, SAT)
    in water-fat MRI images
    """
    # no arguments shows --help option
    if water is None or fat is None:
        with click.Context(cli, info_name='vatsatseg') as ctx:
            click.echo(ctx.get_help())
            return

    # run algorithm
    if not os.path.exists(output):
        vatsatseg(water, fat, output)
    else:
        logger.info('Output %s already exists. -> Open viewer', output)
        show = True

    # open viewer (itk-snap, set path in config.ini)
    if show:
        open_viewer([water, fat], output)

    return


def open_viewer(images, segmentation, labeldescfile=None, cmd=None, opts=None):
    """

    :param images:
    :param segmentation:
    :param labeldesc:
    :param cmd:
    :returns:
    :rtype:

    """
    # read config.ini
    config = configparser.ConfigParser()
    wd = os.path.dirname(__file__)
    config.read(os.path.join(wd, 'config.ini'))
    viewer = config['viewer']

    if labeldescfile is None:
        labeldescfile = os.path.join(wd, viewer.get('labeldescfile'))

    if cmd is None:
        cmd = viewer['cmd']

    if opts is None:
        opts = viewer['opts'].format(*images, segmentation, labeldescfile)

    logger.info('Open viewer: "%s %s".', cmd, opts)
    cmd = ([cmd] + opts.split(' '))
    subprocess.call(cmd)


def vatsatseg(water, fat, output, labeldict=None):
    """perform vatsatseg by reading in filenames for water, fat and output

    :param water: str, filepath to water image file
    :param fat: str, filepath to fat image file
    :param output: str, filepath to output labelmap
    :param labeldict:
    :returns: None
    :rtype:

    """
    logger.info('Read input. ...')
    Water = sitk.ReadImage(water)
    Fat = sitk.ReadImage(fat)

    w_img = sitk.GetArrayFromImage(Water)
    f_img = sitk.GetArrayFromImage(Fat)
    logger.info('Done.')

    if labeldict is None:
        # read config.ini
        config = configparser.ConfigParser()
        wd = os.path.dirname(__file__)
        config.read(os.path.join(wd, 'config.ini'))
        viewer = config['viewer']
        labeldescfile = os.path.join(wd, viewer.get('labeldescfile'))

        logger.info('Read labeldescfile. ...')
        labeldict = read_labeldescfile(labeldescfile)
        logger.info('Done.')

    logger.info('Start segmentation. ...')
    labelmap = vatsatseg_3d(w_img, f_img, labeldict)
    logger.info('Done.')

    label_img = sitk.GetImageFromArray(labelmap.astype(float))
    label_img.CopyInformation(Water)

    sitk.WriteImage(label_img, output)
    logger.info('Wrote labelmap "%s".', output)


def read_labeldescfile(labeldescfile):
    """read ITK label description file (default path set in config.ini)
    and return dictionary of name keys and label integers

    :param labeldescfile: path
    :returns: dictionary of label names and numbers
    :rtype: dict

    """
    labeldict = {}
    with open(labeldescfile, 'r') as f:
        for line in f.readlines():
            entries = [s.lower().replace('"', '')
                       for s in re.split('\s+', line) if s is not '']
            if 'background' in entries:
                labeldict['background'] = entries[0]
            elif 'water' in entries:
                labeldict['water'] = entries[0]
            elif 'vat' in entries:
                labeldict['vat'] = entries[0]
            elif 'sat' in entries:
                labeldict['sat'] = entries[0]
    return labeldict


def vatsatseg_3d(w_img, f_img, labeldict=None):
    """runs vatsatseg on 3d arrays slice by slice

    entry point for the core algorithm: use inpu water and fat arrays as two
    features for a kmeans clustering (k=3: background, water, fat).

    :param w_img: 3d numpy.ndarray
    :param f_img: 3d numpy.ndarray
    :param labeldict:
    :returns: labelmap with 4 components: background, water, vat, sat
    :rtype: 3d numpy.ndarray

    """
    slice_ = np.zeros(w_img.shape[1:])
    labelmap = np.zeros_like(w_img, dtype=int)
    error_slices = []
    for iz in range(w_img.shape[0]):
        try:
            labelmap[iz, :, :] = get_labelmap_2d(w_img[iz, :, :],
                                                 f_img[iz, :, :],
                                                 labeldict)
        except:
            logger.info("Error at slice %s", iz)
            labelmap[iz, :, :] = slice_
            error_slices.append(iz)
    return labelmap


def get_labelmap_2d(w_img, f_img, labeldict=None):
    """after kmeans clustering of background water and fat in one slice
    run logic to differentiate vat and sat and create labelmap

    :param b_mask: 2d numpy.ndarray, binary background mask
    :param w_mask: 2d numpy.ndarray, binary water mask
    :param f_mask: 2d numpy.ndarray, binary fat mask
    :returns: labelmap with 4 components: background 0, water 1, vat 2, sat 3
    :rtype: 2d numpy.ndarray

    """
    # kmeans clustering
    b_mask, w_mask, f_mask = kmeans(w_img, f_img)

    # differentiation between vat and sat
    sat_label, sat_filled, sat_margin = get_intermediate_sat_labels(f_mask)

    skintorso_mask, skin_margin, skin_filled = \
        get_intermediate_skintorso_labels(w_mask, sat_margin, sat_filled)

    torso_mask = get_torso_mask(skin_filled, skintorso_mask)

    arm_mask = get_arm_mask(b_mask, torso_mask)

    torso_water_mask = get_torso_water_mask(w_mask, sat_filled)

    abdominal_water_mask = get_abdominal_water_mask(torso_water_mask)

    vat_mask = get_vat_mask(f_mask, abdominal_water_mask)

    sat_mask = get_sat_mask(sat_label, vat_mask, abdominal_water_mask)

    # create labelmap
    if labeldict is None:
        labeldict = {'background': 0, 'water': 1, 'vat': 2, 'sat': 3,
                     'arms': 0,
                     'abdominal_water': 1,
                     'torso_skin': 1}

    labelmap = np.zeros_like(b_mask, dtype=int)
    labelmap[b_mask.astype(bool)] = labeldict['background']
    labelmap[torso_water_mask.astype(bool)] = labeldict['water']
    labelmap[skintorso_mask.astype(bool)] = \
        labeldict.get('torso_skin', labeldict['water'])
    labelmap[abdominal_water_mask.astype(bool)] = \
        labeldict.get('abdominal_water', labeldict['water'])
    labelmap[vat_mask.astype(bool)] = labeldict['vat']
    labelmap[sat_mask.astype(bool)] = labeldict['sat']
    labelmap[arm_mask.astype(bool)] = \
        labeldict.get('arms', labeldict['background'])

    return labelmap


def kmeans(w_img, f_img):
    """perform k-means clustering with k=3 for water, fat and background
    given input arrays for water and fat,
    input arrays not rescaled (!)

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


def get_intermediate_sat_labels(f_mask):
    """FIXME! briefly describe function

    :param f_mask: 2d numpy.ndarray of bools, fat mask
    :returns: mask of fat label with larges bbox, also filled,
              and its dilation margin
    :rtype: tuple of three 2d numpy.ndarray of bools

    """
    # find sat_label
    labels_fat = label(f_mask)
    regions_fat = regionprops(labels_fat)
    ind_region_sat = np.argmax([r.bbox_area for r in regions_fat
                                if r.label != 0])
    region_sat = regions_fat[ind_region_sat]
    sat_label = labels_fat == region_sat.label # can include some connected vat, correct later to get sat_mask
    # TODO: what to do if sat_label is not continuous?
    # maybe check for a label with a bbox area close to the sat_label?

    # check how thick the watery skin around the torso is at its thinnest location
    sat_filled = binary_fill_holes(sat_label).astype(np.int64)
    # convex maybe hull in case sat is not continuous around the torso
    sat_margin = binary_dilation(sat_filled) ^ sat_filled # initialization to find torso skin

    return sat_label, sat_filled, sat_margin


def get_intermediate_skintorso_labels(w_mask, sat_margin, sat_filled):
    """FIXME! briefly describe function

    :param w_mask: 2d numpy.ndarray, water mask
    :param sat_margin: 2d numpy.ndarray
    :param sat_filled: 2d numpy.ndarray
    :returns:
    :rtype: tuple of three 2d numpy.ndarray of bools

    """
    # find (watery) skin around torso
    # iteratively push skin margin mask to the outside (by binary dilation)
    # if union with the water mask drops by drop_ratio_thresh then stop
    # number of iterations = skinwidth_torso
    skin_filled = sat_filled
    skin_margin = sat_margin
    nvoxels_skin_torso = len(np.where(sat_margin)[0])
    nvoxels_skin_torso_prev = nvoxels_skin_torso
    nvoxel_rel_drop = np.abs(nvoxels_skin_torso - nvoxels_skin_torso_prev) / nvoxels_skin_torso_prev
    drop_ratio_thresh = 0.5
    while nvoxel_rel_drop <= drop_ratio_thresh:
        skin_filled = skin_margin | skin_filled
        skin_margin = (binary_dilation(skin_filled) ^ skin_filled) & w_mask
        nvoxels_skin_torso = nvoxels_skin_torso_prev
        nvoxels_skin_torso_prev = len(np.where(skin_margin)[0])
        if nvoxels_skin_torso_prev == 0:
            break
        nvoxel_rel_drop = np.abs(nvoxels_skin_torso_prev - nvoxels_skin_torso) / nvoxels_skin_torso_prev

    skin_filled = skin_margin | skin_filled
    skintorso_mask = (skin_margin | skin_filled) ^ sat_filled

    return skintorso_mask, skin_margin, skin_filled


def get_torso_mask(skin_filled, skintorso_mask):
    return skin_filled | skintorso_mask


def get_arm_mask(b_mask, torso_mask):
    # can include some torso skin
    return binary_opening(binary_fill_holes(~b_mask & ~torso_mask))


def get_torso_water_mask(w_mask, sat_filled):
    return w_mask & sat_filled


def get_abdominal_water_mask(torso_water_mask):
    """FIXME! briefly describe function

    :param torso_water_mask:
    :returns:
    :rtype:

    """
    labels_torso_water = label(torso_water_mask)
    regions_torso_water = regionprops(labels_torso_water)
    ind_region_torso_water = np.argmax([r.bbox_area for r in regions_torso_water
                                        if r.label != 0])
    vat_region = regions_torso_water[ind_region_torso_water]

    abdominal_water_mask = labels_torso_water == vat_region.label

    return abdominal_water_mask


def get_vat_mask(f_mask, abdominal_water_mask):
    """FIXME! briefly describe function

    :param f_mask:
    :param abdominal_water_mask:
    :returns:
    :rtype:

    """
    # TODO maybe correct convex hull problem
    vat_mask = convex_hull_image(abdominal_water_mask) & f_mask # convex_hull introduced swapped VAT/SAT
    vat_mask = convex_hull_image(binary_opening(vat_mask)) & f_mask
    return vat_mask


def get_sat_mask(sat_label, vat_mask, abdominal_water_mask):
    return sat_label & ~vat_mask & ~abdominal_water_mask


if __name__ == '__main__':

    cli()
