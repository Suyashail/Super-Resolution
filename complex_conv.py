'''
library used to implement Complex-Valued Convolutions and Activations.
Contains complex_conv, complex_transpose_conv, zReLU, modReLU functions.
Works on both real and complex inputs.
Implemeted as defined in the paper-"Deep Complex Networks" and adapted from "Complex-Valued Convolutional Neural Networks
for MRI Reconstruction" by Elizabeth K. Cole et. al; Toolbox for complex-valued convolution and activation
functions using an unrolled architecture."
'''
import tensorflow as tf
import numpy as np
from math import pi

'''
Intuition:
Let X = A+iB and W = X+iY where A,B,C,D are real matrices. Therefore,
X*W = (A+iB)*(X+iY)
    = (A*X - B*Y) + i(A*X + B*Y)
'''
def complex_conv(tf_input, num_features, kernel_size, stride=1,padding='same', data_format="channels_last", use_bias=True,trainable=True):
  num_features = num_features // 2      #divide the total features into half real and half imaginary

  tf_real = tf.math.real(tf_input)
  tf_imag = tf.math.imag(tf_input)

  with tf.compat.v1.variable_scope(None, default_name="complex_conv2d"):
    tf_real_real = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_real)
    tf_imag_real = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_imag)
    tf_real_imag = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_real)
    tf_imag_imag = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_imag)


    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.complex(real_out, imag_out)
    return tf_output

def complex_transposeConv(tf_input, num_features, kernel_size, stride=1,padding='same', data_format="channels_last", use_bias=True,trainable=True):
  num_features = num_features // 2

  tf_real = tf.math.real(tf_input)
  tf_imag = tf.math.imag(tf_input)

  with tf.compat.v1.variable_scope(None, default_name="complex_Tconv2d"):
    tf_real_real = tf.keras.layers.Conv2DTranspose(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_real)
    tf_imag_real = tf.keras.layers.Conv2DTranspose(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_imag)
    tf_real_imag = tf.keras.layers.Conv2DTranspose(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_real)
    tf_imag_imag = tf.keras.layers.Conv2DTranspose(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding=padding, data_format=data_format, trainable=True)(tf_imag)


    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.complex(real_out, imag_out)
    return tf_output

'''
Average Pooling layer takes input a complex tensor and splits it into real and imaginary layers. AvgPool function is applied to both the layers separately. Then the layers are combined as a complex tensor.
'''
def avgpool(tf_input, sizes = None, strides = None, padding = None):
  tf_real = tf.math.real(tf_input)
  tf_imag = tf.math.imag(tf_input)
  real_out = tf.keras.layers.AvgPool2D()(tf_real)
  imag_out = tf.keras.layers.AvgPool2D()(tf_imag)
  tf_output = tf.complex(real_out, imag_out)
  return tf_output


def flatten(tf_input):
  '''
    
  Flattens input array along each dimension if input is high-dimensional algebraic valued
    
  '''
  tf_real = tf.math.real(tf_input)
  tf_imag = tf.math.imag(tf_input)
    
  out = tf.complex(tf.keras.layers.Flatten()(tf_real), tf.keras.layers.Flatten()(tf_imag))
 
  return out

'''
Intuition : Let X = x+iy and W = a+ib
Therefore Y = WX
            = (x+iy)(a+ib)
            = (ax-by) + i(ax+by)
'''
def Dense(tf_input,units,activation = "relu" ):
  tf_real = tf.math.real(tf_input)
  tf_imag = tf.math.imag(tf_input)

  tf_real_real = tf.keras.layers.Dense(units=units, activation = activation)(tf_real)
  tf_imag_real = tf.keras.layers.Dense(units=units, activation = activation)(tf_imag)
  tf_real_imag = tf.keras.layers.Dense(units=units, activation = activation)(tf_real)
  tf_imag_imag = tf.keras.layers.Dense(units=units, activation = activation)(tf_imag)

  real_out = tf_real_real - tf_imag_imag
  imag_out = tf_real_imag - tf_imag_real

  tf_output = tf.complex(real_out,imag_out)
  return tf_output

#Batch Normalisation
def _batch_norm(tf_input, data_format="channels_last", training=False):
    tf_output = tf.compat.v1.layers.batch_normalization(
        tf_input,
        axis=(1 if data_format == "channels_first" else -1),
        training=training,
        renorm=True,
        fused=True,
    )
    return tf_output

#Circular Padding for Circular Convolution Function.
def _circular_pad(tf_input, pad, axis):
    """Perform circular padding. Take elements along axis and pad in front and end"""
    shape_input = tf.shape(tf_input)
    shape_0 = tf.cast(tf.reduce_prod(shape_input[:axis]), dtype=tf.int32)
    shape_axis = shape_input[axis]
    tf_output = tf.reshape(tf_input, tf.stack((shape_0, shape_axis, -1)))

    tf_pre = tf_output[:, shape_axis - pad:, :]
    tf_post = tf_output[:, :pad, :]
    tf_output = tf.concat((tf_pre, tf_output, tf_post), axis=1)

    shape_out = tf.concat(
        (shape_input[:axis], [shape_axis + 2 * pad], shape_input[axis + 1:]), axis=0
    )
    tf_output = tf.reshape(tf_output, shape_out)

    return tf_output


def complex_to_channels(image, name="complex2channels"):
    """Convert data from complex to channels."""
    with tf.name_scope(name):
        image_out = tf.stack([tf.math.real(image), tf.math.imag(image)], axis=-1)
        shape_out = tf.concat(
            [tf.shape(image)[:-1], [image.shape[-1] * 2]], axis=0)
        image_out = tf.reshape(image_out, shape_out)
    return image_out

def channels_to_complex(image, name="channels2complex"):
    """Convert data from channels to complex."""
    with tf.name_scope(name):
        image_out = tf.reshape(image, [-1, 2])
        image_out = tf.complex(image_out[:, 0], image_out[:, 1])
        shape_out = tf.concat(
            [tf.shape(image)[:-1], [image.shape[-1] // 2]], axis=0)
        image_out = tf.reshape(image_out, shape_out)
    return image_out

# Circular Convolution Function
def _conv2d(
    tf_input,
    num_features=128,
    kernel_size=3,
    data_format="channels_last",
    type_conv="real",
    circular=True,
    conjugate=False,
):
    """Conv2d with option for circular convolution."""
    if data_format == "channels_last":
        # (batch, z, y, channels)
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        # (batch, channels, z, y)
        axis_c = 1
        axis_z = 2
        axis_y = 3

    pad = int((kernel_size - 0.5) / 2)
    tf_output = tf_input

    if circular:
        with tf.name_scope("circular_pad"):
            tf_output = _circular_pad(tf_output, pad, axis_z)
            tf_output = _circular_pad(tf_output, pad, axis_y)

    if type_conv == "real":
        print("real convolution")
        num_features = int(num_features) // np.sqrt(2)
        tf_output = tf.keras.layers.conv2d(
            tf_output,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            data_format=data_format,
        )
    if type_conv == "complex":
        print("complex convolution")
        # channels to complex
        #tf_output = tf_util.channels_to_complex(tf_output)

        if num_features != 2:
            num_features = num_features // 2
            
        tf_output = complex_functions.complex_conv(
            tf_output, num_features=num_features, kernel_size=kernel_size)

        if conjugate == True and num_features != 2:
            print("conjugation")
            # conjugate the output
            tf_real = tf.math.real(tf_output)
            imag_out = tf.math.imag(tf_output)
            imag_conj = -1 * imag_out

            real_out = tf.concat([real_out, real_out], axis=-1)
            imag_out = tf.concat([imag_out, imag_conj], axis=-1)

            tf_output = tf.concat([real_out, imag_out], axis=-1)

        # complex to channels
        #tf_output = complex_to_channels(tf_output)

    if circular:
        shape_input = tf.shape(tf_input)
        shape_z = shape_input[axis_z]
        shape_y = shape_input[axis_y]
        with tf.name_scope("circular_crop"):
            if data_format == "channels_last":
                tf_output = tf_output[
                    :, pad: (shape_z + pad), pad: (shape_y + pad), :
                ]
            else:
                tf_output = tf_output[
                    :, :, pad: (shape_z + pad), pad: (shape_y + pad)
                ]
    # add all needed attributes to tensor
    else:
        with tf.name_scope("non_circular"):
            tf_output = tf_output[:, :, :, :]

    return tf_output






  

