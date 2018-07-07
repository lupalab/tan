import copy
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
from ..model import transforms as trans


def make_subsamp_map(level):
    """ Make a subsampling mapping for pixels.
    make_subsamp_map(1) =
        array([[0, 3],
               [2, 1]])

    make_subsamp_map(2) =
        array([[ 0,  3, 12, 15],
               [ 2,  1, 14, 13],
               [ 8, 11,  4,  7],
               [10,  9,  6,  5]])
    ...
    """
    subsamp_map = np.zeros((1, 1), dtype=np.int64)
    for l in range(level):
        factor = 4**l
        subsamp_map = np.concatenate(
            (np.concatenate([subsamp_map, 3*factor+subsamp_map], 1),
             np.concatenate([2*factor+subsamp_map, factor+subsamp_map], 1)),
            0)
    return subsamp_map


def image_subsamp_indices(img_size, subsamp_level, chans=3):
    """ Returns a permutation for dividing pixels into their corresponding
    subsampling groups according to make_subsamp_map(subsmap_level). I.e. for
    subsamp_level = 2 every fourth pixel starting with the top left corner is in
    group 0, every fourth pixel starting at coordinates (1, 1) is in group 1,
    ..., up to group 15.

    Args:
        img_size: int, the dimension of images, is assumed to be power of 2.
        subsamp_level: int, the amount to subsample by, pixels are grouped every
            2**subsamp steps.
        chans: int, number of channels in images.

    Returns:
        squeeze_perm: sub_size x sub_size x chans*nsubgroups int64 array where
            np.reshape(img, (-1))[np.reshape(squeeze_perm, -1)] returns a
            flettened vector of the subsampled pixels, and
            sub_size = (img_size / 2**subsamp_level),
            nsubgroups = 4**subsamp_level.
    """
    # Make image sized mapping of pixel to subsampling group.
    subsamp_map = make_subsamp_map(subsamp_level)
    sub_size = (img_size / 2**subsamp_level)
    subsamp_map = np.concatenate([subsamp_map]*sub_size, 0)
    subsamp_map = np.concatenate([subsamp_map]*sub_size, 1)
    subsamp_map = np.stack([subsamp_map]*chans, -1)
    # Make the squeezing operation index mapping.
    nsubgroups = 4**subsamp_level
    flat_img_inds = np.arange(img_size**2 * chans)
    img_inds = np.reshape(flat_img_inds, (img_size, img_size, chans))
    squeeze_perm = np.concatenate(
        [np.reshape(img_inds[np.equal(subsamp_map, g)],
                    (sub_size, sub_size, chans))
         for g in range(nsubgroups)],
        -1)
    # Make masks for conditioning.
    squeeze_inds = np.zeros_like(squeeze_perm)
    for k in range(nsubgroups):
        squeeze_inds[:, :, k*chans:(k+1)*chans] = k
    masks = [np.float32(np.less(squeeze_inds, g)) for g in range(nsubgroups)]
    return squeeze_perm, masks


def subsamp_image(images, levels=1, return_masks=False):
    img_shape = images.get_shape().as_list()
    img_size = img_shape[1]
    chans = img_shape[-1]
    # Get permutation for subsampling.
    subsamp_perm, subsamp_masks = image_subsamp_indices(img_size, levels, chans)
    # Permute and reshape.
    subsamp_flat = np.reshape(subsamp_perm, (-1,))
    imgs_flat = tf.reshape(images, (-1, np.prod(img_shape[1:])))
    # imgs_sub = tf.reshape(imgs_flat[:, subsamp_flat], [-1]+subsamp_perm.shape)
    # TODO: Better way to shuffle?
    imgs_sub = tf.reshape(
        tf.transpose(tf.gather(tf.transpose(imgs_flat), subsamp_flat)),
        [-1]+[s for s in subsamp_perm.shape])

    # Turn subsampled back into original dimensions.
    def resamp_image(sub_images):
        sub_img_shape = images.get_shape().as_list()
        sub_imgs_flat = tf.reshape(sub_images, (-1, np.prod(sub_img_shape[1:])))
        inv_subsamp_flat = trans.invperm(subsamp_flat)
        # TODO: Better way to shuffle?
        # return tf.reshape(sub_imgs_flat[:, inv_subsamp_flat],
        #                   [-1]+img_shape)
        return tf.reshape(
            tf.transpose(tf.gather(tf.transpose(sub_imgs_flat),
                                   inv_subsamp_flat)),
            [-1]+img_shape[1:])
    if return_masks:
        return imgs_sub, resamp_image, subsamp_masks
    return imgs_sub, resamp_image


def rgb2bw(image, name=None):
    ndims = len(image.get_shape())
    if ndims == 3:
        bw_weights_np = np.zeros((1, 1, 3), np.float32)
        bw_weights_np[0, 0, :] = [0.299, 0.587, 0.114]
    else:
        bw_weights_np = np.zeros((1, 1, 1, 3), np.float32)
        bw_weights_np[0, 0, 0, :] = [0.299, 0.587, 0.114]
    bw_weights = tf.constant(bw_weights_np)
    return tf.reduce_sum(image*bw_weights, -1, keep_dims=True, name=name)


def get_image(filename, img_size=(64, 64, 3), downsample=1, do_resize=True,
              center_crop=None, do_bw=True, do_logit=True, noise_scale=None):
    def logit(x):
        return tf.log(x) - tf.log(1.0-x)

    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, img_size[-1], downsample)
    if center_crop is not None:
        imshape = tf.shape(image)
        j = (imshape[0]-center_crop)/2
        i = (imshape[1]-center_crop)/2
        image = tf.image.crop_to_bounding_box(
            image, j, i, center_crop, center_crop
        )
    if do_resize:
        image = tf.image.resize_images(image, img_size[:2])
    else:
        image.set_shape(img_size)
    image = tf.cast(image, tf.float32)
    if do_bw:
        image = rgb2bw(image)
    if noise_scale is None:  # defualt noise of 0.5 unif
        noise_scale = 0.5
    if not isinstance(noise_scale, float) or noise_scale > 0.0:
        print('Using noise with scale {}'.format(noise_scale))
        noise = tf.random_uniform(image.get_shape())
        image = image + noise_scale*noise
    if do_logit:
        image = logit(0.05 + (1.0-0.05)*image/256.0)
    return image


def save_block(images, save_path, nimgs=30, truncate=False, logit=False,
               minval=0.0, maxval=1.0, img_size=64, chans=3, permute=True):
    """ Helper function to visualize image samples stiched together. """
    if permute:
        perm = np.random.permutation(len(images))
    else:
        perm = np.arange(len(images))
    if chans == 1:
        images_block = np.zeros((img_size*nimgs, img_size*nimgs))
    else:
        images_block = np.zeros((img_size*nimgs, img_size*nimgs, chans))
    for i in range(nimgs):
        for j in range(nimgs):
            k = i*nimgs+j
            img = copy.copy(images[perm[k]])
            if truncate:
                img[img < minval] = minval
                img[img > maxval] = maxval
            if logit:
                img = 1.0/(1.0+np.exp(-img))
            if chans == 1:
                img = np.reshape(img, (img_size, img_size))
                images_block[i*img_size:(i+1)*img_size,
                             j*img_size:(j+1)*img_size] = img
            else:
                img = np.reshape(img, (img_size, img_size, chans))
                images_block[i*img_size:(i+1)*img_size,
                             j*img_size:(j+1)*img_size, :] = img
    plt.imsave(save_path, images_block, cmap=plt.cm.gray)
