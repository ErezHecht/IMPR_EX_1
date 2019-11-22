import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

# ========================= Constants =========================
MAX_INTENSITY = 255
YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.275, -0.321],
                       [0.212, -0.523, 0.311]])
INV_YIQ_MATRIX = np.linalg.inv(YIQ_MATRIX)
GS_REPRESENTATION = 1
RGB_REPRESENTATION = 2
REPRESENTATIONS = [GS_REPRESENTATION, RGB_REPRESENTATION]


# ========================= Reading and Displaying =========================

def read_image(filename, representation):
    """
    Reads a given image file and converts it into the given representation.
    :param filename: the filename of an image on the disk (grayscale or RGB)
    :param representation: 1 or 2, where 1 = grayscale, 2 = RGB.
    :return: An image represented by a matrix of type np.float64 with
    intensities normalized to the range [0,1].
    """
    if not os.path.isfile(filename):
        return
    if representation not in REPRESENTATIONS:
        return
    im = np.array(imread(filename))
    # Assumes if type is float64, range is 0-1:
    out_im = im
    # Case: Type is uint8: range is 0-255
    if im.dtype == np.uint8:
        out_im = _get_image_as_float(im)
    # Assumes original image is not given in grayscale and representation = 2
    if representation == GS_REPRESENTATION:
        out_im = rgb2gray(out_im)
    return out_im


def imdisplay(filename, representation):
    """
    Displays an image in a given representation.
    :param filename: the filename of an image on the disk (grayscale or RGB)
    :param representation: 1 or 2, where 1 = grayscale, 2 = RGB.
    """
    out_im = read_image(filename, representation)
    if out_im is None:
        return
    if representation == GS_REPRESENTATION:
        plt.imshow(out_im, cmap=plt.get_cmap("gray"))
    if representation == RGB_REPRESENTATION:
        plt.imshow(out_im)
    plt.show()


# ========================= Color Space Conversion =========================

def rgb2yiq(im_yiq):
    """
    Transforms a RGB image into the YIQ color space.
    :param im_yiq: a np.float64 matrix of size height*width*3
    :return: an image with the same dimensions as imRGB, converted to the
    YIQ color space
    """
    result = np.dot(im_yiq, YIQ_MATRIX.T)
    return result


def yiq2rgb(im_yiq):
    """
    Transforms a YIQ image into the RGB color space.
    :param im_yiq: a np.float64 matrix of size height*width*3
    :return: an image with the same dimensions as imYIQ, converted to the
    RGB color space
    """
    result = np.dot(im_yiq, INV_YIQ_MATRIX.T)
    return result


# ========================= Histogram Equalization =========================
def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given RGB or GS image.
    :param im_orig: RGB or GS float64 image with values in [0,1].
    :return: a list [im_eq, hist_orig, hist_eq] where:
            im_eq: the equalized image - RGB or GS float64 image with values
            in [0,1].
            hist_orig: 256 histogram of the original image
            (array with size 256)
            hist_eq: 256 histogram of the equalized image
            (array with size 256)
    """
    im_orig_arr = np.array(im_orig).astype(np.float64)
    is_rgb = len(im_orig_arr.shape) == 3
    # Case: Image is RGB - Convert to YIQ and manipulate only Y:
    if is_rgb:
        im_orig_arr = rgb2yiq(im_orig)[:, :, 0]
    im_orig_arr = _get_image_as_int(im_orig_arr)
    pixel_num = im_orig_arr.size
    # Histogram for original image:
    hist_orig, bounds = np.histogram(im_orig_arr, bins=256,
                                     range=[0, MAX_INTENSITY])
    # Cumulative Histogram:
    cumulative_hist = hist_orig.cumsum()
    # Get C(m), where C is the cumulative histogram, and m is the index of the
    # first non zero value
    first_non_zero_value = cumulative_hist[cumulative_hist != 0][0]
    stretched_cumulative_histogram = cumulative_hist - first_non_zero_value
    stretch_correction = pixel_num - first_non_zero_value
    # Normalized Cumulative Histogram:
    normal_cumulative_hist = (stretched_cumulative_histogram /
                              stretch_correction) * MAX_INTENSITY
    # Round numbers:
    normal_cumulative_hist = normal_cumulative_hist.astype(np.uint8)
    im_eq = _apply_transformation(im_orig, im_orig_arr, normal_cumulative_hist,
                                  is_rgb, bounds)
    hist_eq = np.histogram(im_eq, bins=MAX_INTENSITY + 1,
                           range=[0, MAX_INTENSITY])[0]
    im_eq = np.clip(im_eq, 0, 1)
    return im_eq, hist_orig, hist_eq


# ========================= Image Quantization =========================

def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given RGB or GS image.
    :param im_orig: GS or RGB image to be quantized (float64 image with
    values [0,1])
    :param n_quant: Number of intensities the output will have
    :param n_iter: maximum number of iterations of the optimization procedure
    :return: a list [im_quant, error] where:
            im_quant: the quantized image
            error: array with size (n_iter) or less of the total intensities
            errors for each iteration of the quantization procedure.return
    """
    is_rgb = len(im_orig.shape) == 3
    im_orig_arr = np.copy(im_orig)
    # If RGB image, work on Y channel of YIQ converted image
    if is_rgb:
        im_orig_arr = rgb2yiq(im_orig)[:, :, 0]
    hist_orig, bounds = np.histogram(im_orig_arr, bins=256,
                                     range=[0, 1])

    cumu_hist_orig = hist_orig.cumsum()
    pixel_num = cumu_hist_orig[-1]
    partition_size = pixel_num / n_quant
    # Find initial Z values: the index of the maximal value in each partition
    # to achieve a balanced spread of pixels
    z_indices = [0]
    for i in range(1, n_quant + 1):
        z_indices.append(np.argmax(cumu_hist_orig[cumu_hist_orig <=
                                                  partition_size * i]))

    z_indices = np.rint(z_indices).astype(np.uint8)
    q_values = [0] * n_quant
    error = []
    for j in range(n_iter):
        # Update q values:
        for i in range(n_quant):
            q_values[i] = _calculate_q_value(hist_orig, z_indices, i)
        # Update error
        error.append(_calculate_error(hist_orig, z_indices, q_values, n_quant))
        old_z = z_indices.copy()
        # Update z indices
        for i in range(1, n_quant):
            z_indices[i] = ((q_values[i] + q_values[i - 1]) / 2)
        z_indices = np.rint(z_indices).astype(np.uint8)
        if z_indices.all() == old_z.all():
            break
    # Concentrate segment pixels to q values:
    quant_hist = hist_orig.copy()
    for i in range(n_quant):
        indices_range = np.arange(z_indices[i], z_indices[i + 1] + 1)
        quant_hist[indices_range] = q_values[i]
    im_quant = _apply_transformation(im_orig, im_orig_arr, quant_hist,
                                     is_rgb, bounds)
    return im_quant, error


# ========================= Helper Functions =========================

def _apply_transformation(im_orig, im_orig_arr, transform_histogram, is_rgb,
                          bounds):
    """
    Applies a LUT transformation to a given image
    :param im_orig: Original image
    :param im_orig_arr: Np array of the original image after conversions
    :param transform_histogram: the histogram to apply to the LUT
    :param is_rgb: True if the original image is an RGB image, false otherwise
    :param bounds: the bins of the histogram
    :return: An image after the LUT was updated
    """
    transform_im = np.interp(im_orig_arr, bounds[:-1], transform_histogram)
    transform_im = _get_image_as_float(transform_im)
    if is_rgb:
        yiq_im_orig = rgb2yiq(im_orig)
        yiq_im_orig[:, :, 0] = transform_im
        transform_im = yiq2rgb(yiq_im_orig)
    return transform_im


def _calculate_error(hist_orig, z_indices, q_values, n_quant):
    """
    Calculates the error in the quantization process
    :param hist_orig: the original histogram
    :param z_indices: an array of z indices
    :param q_values: an array of q values
    :param n_quant: the number of colors given in the quantization process
    :return: the error
    """
    error = 0
    for i in range(n_quant):
        indices_range = np.arange(z_indices[i], z_indices[i + 1] + 1)
        partition_hist = hist_orig[indices_range]
        q_i = q_values[i]
        indices_range -= int(q_i)
        indices_range **= 2

        error += np.dot(indices_range, partition_hist.T)
    return error


def _calculate_q_value(hist_orig, z_indices, index):
    """
    Calculates the q value
    :param hist_orig: The histogram
    :param z_indices: Z index array
    :param index: the index of the segment
    :return: the q value
    """
    indices_range = np.arange(z_indices[index], z_indices[index + 1] + 1)
    partition_hist = hist_orig[indices_range]
    q_i = np.dot(indices_range,
                 partition_hist.T) / partition_hist.sum()
    return q_i


def _get_image_as_int(float_im):
    """
    Gets a numpy array representing the given image in the range [0,255]
    :param float_im: A float64 numpy array with values in the range [0,1]
    :return: an uint8 numpy array with values in the range [0,255]
    """
    int_im = float_im * MAX_INTENSITY
    return int_im.astype(np.uint8)


def _get_image_as_float(int_im):
    """
    Gets a numpy array representing the given image in the range [0,1]
    :param int_im: A uint8 numpy array with values in the range [0,255]
    :return: an float64 numpy array with values in the range [0,1]
    """
    return int_im.astype(np.float64) / MAX_INTENSITY
