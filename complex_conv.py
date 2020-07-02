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
def complex_conv(tf_input, num_features, kernel_size, stride=1, data_format="channels_last", use_bias=True,trainable=True):
  num_features = num_features // 2      #divide the total features into half real and half imaginary

  tf_real = tf.math.real(tf_input)
  tf_imag = tf.math.imag(tf_input)

  with tf.compat.v1.variable_scope(None, default_name="complex_conv2d"):
    tf_real_real = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_real)
    tf_imag_real = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_imag)
    tf_real_imag = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_real)
    tf_imag_imag = tf.keras.layers.Conv2D(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_imag)


    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.complex(real_out, imag_out)
    return tf_output

def complex_transposeConv(tf_input, num_features, kernel_size, stride=1, data_format="channels_last", use_bias=True,trainable=True):
  num_features = num_features // 2

  tf_real = tf.math.real(tf_input)
  tf_imag = tf.math.imag(tf_input)

  with tf.compat.v1.variable_scope(None, default_name="complex_conv2d"):
    tf_real_real = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_real)
    tf_imag_real = tf.keras.layers.Conv2DTranspose(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_imag)
    tf_real_imag = tf.keras.layers.Conv2DTranspose(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_real)
    tf_imag_imag = tf.keras.layers.Conv2DTranspose(filters=num_features,kernel_size=kernel_size, strides=(stride,stride), padding='same', data_format=data_format, trainable=True)(tf_imag)


    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.complex(real_out, imag_out)
    return tf_output

def zrelu(x):
    # x and tf_output are complex-valued
    phase = tf.math.angle(x)

    # Check whether phase <= pi/2
    le = tf.less_equal(phase, pi / 2)

    # if phase <= pi/2, keep it in comp
    # if phase > pi/2, throw it away and set comp equal to 0
    y = tf.zeros_like(x)
    x = tf.where(le, x, y)

    # Check whether phase >= 0
    ge = tf.greater_equal(phase, 0)

    # if phase >= 0, keep it
    # if phase < 0, throw it away and set output equal to 0
    output = tf.where(ge, x, y)

    return output

def modrelu(x, data_format="channels_last"):
    input_shape = tf.shape(x)
    if data_format == "channels_last":
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3

    # Channel size
    shape_c = x.shape[axis_c]

    with tf.name_scope("bias") as scope:
        if data_format == "channels_last":
            bias_shape = (1, 1, 1, shape_c)
        else:
            bias_shape = (1, shape_c, 1, 1)
        bias = tf.compat.v1.get_variable(name=scope,
                               shape=bias_shape,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
    # relu(|z|+b) * (z / |z|)
    norm = tf.abs(x)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    output = tf.complex(tf.math.real(x) * scale,
                        tf.math.imag(x) * scale)

    return output
