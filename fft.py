""" FFT and IFFT Functions """
import tensorflow as tf
import numpy as np
from math import pi



def fftshift(im, axis=0, name="fftshift"):
    """Perform fft shift.
    This function assumes that the axis to perform fftshift is divisible by 2.
    """
    with tf.name_scope(name):
        split0, split1 = tf.split(im, 2, axis=axis)
        output = tf.concat((split1, split0), axis=axis)

    return output


def ifftc(im, name="ifftc", do_orthonorm=True):
    """Centered iFFT on second to last dimension."""
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
        with tf.device("/gpu:0"):
            # FFT is only supported on the GPU
            im_out = tf.signal.ifft(im_out) * fftscale
        if len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out


def fftc(im, name="fftc", do_orthonorm=True):
    """Centered FFT on second to last dimension."""
    with tf.name_scope(name):
        im_out = im
        if do_orthonorm:
            fftscale = tf.sqrt(1.0 * im_out.get_shape().as_list()[-2])
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)
        if len(im.get_shape()) == 4:
            im_out = tf.transpose(im_out, [0, 3, 1, 2])
            im_out = fftshift(im_out, axis=3)
        else:
            im_out = tf.transpose(im_out, [2, 0, 1])
            im_out = fftshift(im_out, axis=2)
        with tf.device("/gpu:0"):
            im_out = tf.signal.fft(im_out) / fftscale
        if len(im.get_shape()) == 4:
            im_out = fftshift(im_out, axis=3)
            im_out = tf.transpose(im_out, [0, 2, 3, 1])
        else:
            im_out = fftshift(im_out, axis=2)
            im_out = tf.transpose(im_out, [1, 2, 0])

    return im_out

def ifft2c(im, name="ifft2c", do_orthonorm=True):
    """Centered inverse FFT2 on second and third dimensions."""
    with tf.name_scope(name):
        im_out = im
        dims = tf.shape(im_out)
        if do_orthonorm:
            fftscale = tf.sqrt(tf.cast(dims[1] * dims[2], dtype=tf.float32))
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        # permute FFT dimensions to be the last (faster!)
        tpdims = list(range(len(im_out.get_shape().as_list())))
        tpdims[-1], tpdims[1] = tpdims[1], tpdims[-1]
        tpdims[-2], tpdims[2] = tpdims[2], tpdims[-2]

        im_out = tf.transpose(im_out, tpdims)
        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)

        # with tf.device('/gpu:0'):
        im_out = tf.signal.ifft2d(im_out) * fftscale

        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)
        im_out = tf.transpose(im_out, tpdims)

    return im_out


def fft2c(im, name="fft2c", do_orthonorm=True):
    """Centered FFT2 on second and third dimensions."""
    with tf.name_scope(name):
        im_out = im
        dims = tf.shape(im_out)
        if do_orthonorm:
            fftscale = tf.sqrt(tf.cast(dims[1] * dims[2], dtype=tf.float32))
        else:
            fftscale = 1.0
        fftscale = tf.cast(fftscale, dtype=tf.complex64)

        # permute FFT dimensions to be the last (faster!)
        tpdims = list(range(len(im_out.get_shape().as_list())))
        tpdims[-1], tpdims[1] = tpdims[1], tpdims[-1]
        tpdims[-2], tpdims[2] = tpdims[2], tpdims[-2]

        im_out = tf.transpose(im_out, tpdims)
        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)

        # with tf.device('/gpu:0'):
        im_out = tf.signal.fft2d(im_out) / fftscale

        im_out = fftshift(im_out, axis=-1)
        im_out = fftshift(im_out, axis=-2)
        im_out = tf.transpose(im_out, tpdims)

    return im_out
