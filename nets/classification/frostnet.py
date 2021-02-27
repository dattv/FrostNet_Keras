"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


class ConvBNReLU_layer(tf.keras.layers.Layer):
    """

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, initializer="he_normal", **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        """
        super(ConvBNReLU_layer, self).__init__()

        self.pad = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))
        self.initializer = initializer

        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding='valid',
            dilation_rate=dilation, use_bias=False,
            groups=groups
        )

        self.bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """

        outputs = self.pad(inputs)
        outputs = self.conv(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        return outputs

    def get_config(self):
        """

        :return:
        """
        base_config = super(ConvBNReLU_layer, self).get_config()

        config = {"initializer": self.initializer }
        return dict(list(base_config.items()) + list(config.items()))


class ConvReLU_layer(tf.keras.layers.Layer):
    """

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, initializer="he_normal", **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        """
        super(ConvReLU_layer, self).__init__()
        self.pad = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))

        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding='valid',
            dilation_rate=dilation, use_bias=False,
            groups=groups
        )

        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        outputs = self.pad(inputs)
        outputs = self.conv(outputs)
        outputs = self.relu(outputs)

        return outputs

    def get_config(self):
        """

        :return:
        """
        base_config = super(ConvReLU_layer, self).get_config()

        config = {"initializer": self.initializer}
        return dict(list(base_config.items()) + list(config.items()))


class ConvBN_layer(tf.keras.layers.Layer):
    """

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, initializer="he_normal", **kwargs):
        """
        
        :param in_channels: 
        :param out_channels: 
        :param kernel_size: 
        :param stride: 
        :param padding: 
        :param dilation: 
        :param groups: 
        :param initializer: 
        :param kwargs: 
        """
        super(ConvBN_layer, self).__init__()

        self.pad = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))

        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding='valid',
            dilation_rate=dilation, use_bias=False,
            groups=groups
        )

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        """
        
        :param inputs: 
        :param kwargs: 
        :return: 
        """
        outputs = self.pad(inputs)
        outputs = self.conv(outputs)
        outputs = self.bn(outputs)
        return outputs

    def get_config(self):
        """
        
        :return: 
        """
        base_config = super(ConvBN_layer, self).get_config()

        config = {"initializer": self.initializer}
        return dict(list(base_config.items()) + list(config.items()))


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CascadePreExBottleneck_layer(tf.keras.layers.Layer):
    """

    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, expand_ratio=6,
                 reduce_factor=4, block_type='CAS'):
        """

        :param in_channels:
        :param out_channels:
        :param quantized:
        :param kernel_size:
        :param stride:
        :param dilation:
        :param expand_ratio:
        :param reduce_factor:
        :param block_type:
        """
        super(CascadePreExBottleneck_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        if in_channels // reduce_factor < 8:
            block_type = 'MB'
        self.block_type = block_type

        r_channels = _make_divisible(in_channels // reduce_factor)

        if stride == 1 and in_channels == out_channels:
            self.reduction = False
        else:
            self.reduction = True

        if self.expand_ratio == 1:
            self.squeeze_conv = None
            self.conv1 = None
            n_channels = in_channels
        else:
            if block_type == 'CAS':
                self.squeeze_conv = ConvBNReLU_layer(in_channels, r_channels, 1, dilation=dilation)
                n_channels = r_channels + in_channels
            else:
                n_channels = in_channels
            self.conv1 = ConvBNReLU_layer(n_channels, n_channels * expand_ratio, 1, dilation=dilation)

        self.conv2 = ConvBNReLU_layer(n_channels * expand_ratio, n_channels * expand_ratio, kernel_size, stride,
                                      (kernel_size - 1) // 2, 1,
                                      groups=n_channels * expand_ratio)
        self.reduce_conv = ConvBN_layer(n_channels * expand_ratio, out_channels, 1, dilation=dilation)

    def call(self, inputs, training=None, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """

        if not self.expand_ratio == 1:
            if self.block_type == 'CAS':
                squeezed = self.squeeze_conv(inputs)
                out = tf.keras.layers.Concatenate()([squeezed, inputs])
            else:
                out = inputs
            out = self.conv1(out)
        else:
            out = inputs
        out = self.conv2(out)
        out = self.reduce_conv(out)

        if not self.reduction:
            out = tf.keras.layers.Add()([inputs, out])
        return out

    def get_config(self):
        """

        :return:
        """
        base_config = super(CascadePreExBottleneck_layer, self).get_config()

        config = {"initializer": self.initializer}
        return dict(list(base_config.items()) + list(config.items()))


class make_layer(tf.keras.layers.Layer):
    """"
    """

    def __init__(self, in_channels, block, block_setting, width_mult, dilation=1, initializer="he_normal"):
        """

        :param block:
        :param block_setting:
        :param width_mult:
        :param dilation:
        """
        super(make_layer, self).__init__()
        self.initializer = initializer
        self.layers = list()
        for k, c, e, r, s in block_setting:
            out_channels = _make_divisible(int(c * width_mult))
            stride = s if (dilation == 1) else 1
            self.layers.append(
                block(in_channels, out_channels, kernel_size=k,
                      stride=s, dilation=dilation, expand_ratio=e, reduce_factor=r)
            )
            in_channels = out_channels

        self._init_set_name('frost_bottleneck_{}_{}'.format(in_channels, width_mult))

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)

        return inputs

    def get_config(self):
        """

        :return:
        """
        base_config = super(make_layer, self).get_config()

        config = {"initializer": self.initializer}
        return dict(list(base_config.items()) + list(config.items()))


class FrostNet_model_class(tf.keras.Model):
    """

    """

    def __init__(self, nclass=1000, mode='large', width_mult=1.0,
                 bottleneck=CascadePreExBottleneck_layer, drop_rate=0.2, dilated=False, initializer="he_normal",
                 **kwargs):
        """

        :param nclass:
        :param mode:
        :param width_mult:
        :param bottleneck:
        :param drop_rate:
        :param dilated:
        :param kwargs:
        """
        super(FrostNet_model_class, self).__init__()

        if mode == 'large':
            layer1_setting = [
                # kernel_size, c, e, r, s
                [3, 16, 1, 1, 1],  # 0
                [3, 24, 6, 4, 2],  # 1
                # [, , , , ],      #2
                # [, , , , ],      #3
                [3, 24, 3, 4, 1],  # 4
            ]
            layer2_setting = [
                [5, 40, 6, 4, 2],  # 5
                # [, , , , ],      #6
                # [, , , , ],      #7
                [3, 40, 3, 4, 1],  # 8

            ]

            layer3_setting = [
                [5, 80, 6, 4, 2],  # 9
                # [, , , , ],      #10
                [5, 80, 3, 4, 1],  # 11
                [5, 80, 3, 4, 1],  # 12

                [5, 96, 6, 4, 1],  # 13
                # [, , , , ],      #14
                [5, 96, 3, 4, 1],  # 15
                [3, 96, 3, 4, 1],  # 16
                [3, 96, 3, 4, 1],  # 17
            ]

            layer4_setting = [
                [5, 192, 6, 2, 2],  # 18
                [5, 192, 6, 4, 1],  # 19
                [5, 192, 6, 4, 1],  # 20
                [5, 192, 3, 4, 1],  # 21
                [5, 192, 3, 4, 1],  # 22
            ]

            layer5_setting = [
                [5, 320, 6, 2, 1],  # 23
            ]

        elif mode == 'base':
            layer1_setting = [
                # kernel_size, c, e, r, s
                [3, 16, 1, 1, 1],  # 0
                [5, 24, 6, 4, 2],  # 1
                # [, , , , ],      #2
                # [, , , , ],      #3
                [3, 24, 3, 4, 1],  # 4
            ]
            layer2_setting = [
                [5, 40, 3, 4, 2],  # 5
                # [, , , , ],      #6
                [5, 40, 3, 4, 1],  # 7
                # [, , , , ],      #8
            ]

            layer3_setting = [
                [5, 80, 3, 4, 2],  # 9
                # [, , , , ],      #10
                # [, , , , ],      #11
                [3, 80, 3, 4, 1],  # 12

                [5, 96, 3, 2, 1],  # 13
                [3, 96, 3, 4, 1],  # 14
                [5, 96, 3, 4, 1],  # 15
                [5, 96, 3, 4, 1],  # 16
            ]

            layer4_setting = [
                [5, 192, 6, 2, 2],  # 17
                [5, 192, 3, 2, 1],  # 18
                [5, 192, 3, 2, 1],  # 19
                [5, 192, 3, 2, 1],  # 20
            ]

            layer5_setting = [
                [5, 320, 6, 2, 1],  # 21
            ]

        elif mode == 'small':
            layer1_setting = [
                # kernel_size, c, e, r, s
                [3, 16, 1, 1, 1],  # 0
                [5, 24, 3, 4, 2],  # 1
                [3, 24, 3, 4, 1],  # 2
                # [, , , , ],      #3
            ]
            layer2_setting = [
                [5, 40, 3, 4, 2],  # 4
                # [, , , , ],      #5
                # [, , , , ],      #6
            ]

            layer3_setting = [
                [5, 80, 3, 4, 2],  # 7
                [5, 80, 3, 4, 1],  # 8
                [3, 80, 3, 4, 1],  # 9

                [5, 96, 3, 2, 1],  # 10
                [5, 96, 3, 4, 1],  # 11
                [5, 96, 3, 4, 1],  # 12
            ]

            layer4_setting = [
                [5, 192, 6, 4, 2],  # 13
                [5, 192, 6, 4, 1],  # 14
                [5, 192, 6, 4, 1],  # 15
            ]

            layer5_setting = [
                [5, 320, 6, 2, 1],  # 16
            ]
        else:
            raise ValueError('Unknown mode.')
        # building first layer

        self.in_channels = _make_divisible(int(32 * min(1.0, width_mult)))

        self.conv1 = ConvBNReLU_layer(3, self.in_channels, 3, 2, 1, initializer=initializer)

        # building bottleneck blocks
        self.layer1 = make_layer(self.in_channels, bottleneck, layer1_setting, width_mult, 1)
        self.layer2 = make_layer(self.in_channels, bottleneck, layer2_setting, width_mult, 1)
        self.layer3 = make_layer(self.in_channels, bottleneck, layer3_setting, width_mult, 1)
        if dilated:
            dilation = 2
        else:
            dilation = 1
        self.layer4 = make_layer(self.in_channels, bottleneck, layer4_setting, width_mult, dilation)
        self.layer5 = make_layer(self.in_channels, bottleneck, layer5_setting, width_mult, dilation)

        # building last several layers
        last_in_channels = self.in_channels

        self.last_layer = ConvBNReLU_layer(last_in_channels, 1280, 1, initializer=initializer)

        self.Dropout = tf.keras.layers.Dropout(rate=drop_rate)
        self.conv2d = tf.keras.layers.Conv2D(filters=nclass, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
                                             kernel_initializer=tf.keras.initializers(initializer))

        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Dropout(drop_rate),
        #     nn.Conv2d(1280, nclass, 1)
        # )

        self.mode = mode

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        x = inputs
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_layer(x)
        # x = self.classifier(x)
        x_shape = x.shape.as_list()
        x = tf.keras.layers.AveragePooling2D(
            pool_size=(x_shape[1], x_shape[2]), strides=(x_shape[1], x_shape[2])
        )(x)
        x = self.Dropout(x)
        x = self.conv2d(x)

        return tf.keras.layers.Flatten()(x)


def FrostNet_inference(input_shape,
                       training=True,
                       nclass=1000,
                       mode='large',
                       width_mult=1.0,
                       bottleneck=CascadePreExBottleneck_layer,
                       drop_rate=0.2,
                       dilated=False,
                       initializer="he_normal"
                       ):
    """

    :param input_shape:
    :param training:
    :param nclass:
    :param mode:
    :param width_mult:
    :param bottleneck:
    :param drop_rate:
    :param dilated:
    :param initializer:
    :return:
    """
    if mode == 'large':
        layer1_setting = [
            # kernel_size, c, e, r, s
            [3, 16, 1, 1, 1],  # 0
            [3, 24, 6, 4, 2],  # 1
            # [, , , , ],      #2
            # [, , , , ],      #3
            [3, 24, 3, 4, 1],  # 4
        ]
        layer2_setting = [
            [5, 40, 6, 4, 2],  # 5
            # [, , , , ],      #6
            # [, , , , ],      #7
            [3, 40, 3, 4, 1],  # 8

        ]

        layer3_setting = [
            [5, 80, 6, 4, 2],  # 9
            # [, , , , ],      #10
            [5, 80, 3, 4, 1],  # 11
            [5, 80, 3, 4, 1],  # 12

            [5, 96, 6, 4, 1],  # 13
            # [, , , , ],      #14
            [5, 96, 3, 4, 1],  # 15
            [3, 96, 3, 4, 1],  # 16
            [3, 96, 3, 4, 1],  # 17
        ]

        layer4_setting = [
            [5, 192, 6, 2, 2],  # 18
            [5, 192, 6, 4, 1],  # 19
            [5, 192, 6, 4, 1],  # 20
            [5, 192, 3, 4, 1],  # 21
            [5, 192, 3, 4, 1],  # 22
        ]

        layer5_setting = [
            [5, 320, 6, 2, 1],  # 23
        ]

    elif mode == 'base':
        layer1_setting = [
            # kernel_size, c, e, r, s
            [3, 16, 1, 1, 1],  # 0
            [5, 24, 6, 4, 2],  # 1
            # [, , , , ],      #2
            # [, , , , ],      #3
            [3, 24, 3, 4, 1],  # 4
        ]
        layer2_setting = [
            [5, 40, 3, 4, 2],  # 5
            # [, , , , ],      #6
            [5, 40, 3, 4, 1],  # 7
            # [, , , , ],      #8
        ]

        layer3_setting = [
            [5, 80, 3, 4, 2],  # 9
            # [, , , , ],      #10
            # [, , , , ],      #11
            [3, 80, 3, 4, 1],  # 12

            [5, 96, 3, 2, 1],  # 13
            [3, 96, 3, 4, 1],  # 14
            [5, 96, 3, 4, 1],  # 15
            [5, 96, 3, 4, 1],  # 16
        ]

        layer4_setting = [
            [5, 192, 6, 2, 2],  # 17
            [5, 192, 3, 2, 1],  # 18
            [5, 192, 3, 2, 1],  # 19
            [5, 192, 3, 2, 1],  # 20
        ]

        layer5_setting = [
            [5, 320, 6, 2, 1],  # 21
        ]

    elif mode == 'small':
        layer1_setting = [
            # kernel_size, c, e, r, s
            [3, 16, 1, 1, 1],  # 0
            [5, 24, 3, 4, 2],  # 1
            [3, 24, 3, 4, 1],  # 2
            # [, , , , ],      #3
        ]
        layer2_setting = [
            [5, 40, 3, 4, 2],  # 4
            # [, , , , ],      #5
            # [, , , , ],      #6
        ]

        layer3_setting = [
            [5, 80, 3, 4, 2],  # 7
            [5, 80, 3, 4, 1],  # 8
            [3, 80, 3, 4, 1],  # 9

            [5, 96, 3, 2, 1],  # 10
            [5, 96, 3, 4, 1],  # 11
            [5, 96, 3, 4, 1],  # 12
        ]

        layer4_setting = [
            [5, 192, 6, 4, 2],  # 13
            [5, 192, 6, 4, 1],  # 14
            [5, 192, 6, 4, 1],  # 15
        ]

        layer5_setting = [
            [5, 320, 6, 2, 1],  # 16
        ]
    else:
        raise ValueError('Unknown mode.')

    input_layer = tf.keras.Input(shape=input_shape)

    in_channels = _make_divisible(int(32 * min(1.0, width_mult)))

    conv1 = ConvBNReLU_layer(3, in_channels, 3, 2, 1, initializer=initializer)(input_layer)

    # building bottleneck blocks
    layer1 = make_layer(in_channels, bottleneck, layer1_setting, width_mult, 1)(conv1)
    layer2 = make_layer(in_channels, bottleneck, layer2_setting, width_mult, 1)(layer1)
    layer3 = make_layer(in_channels, bottleneck, layer3_setting, width_mult, 1)(layer2)
    if dilated:
        dilation = 2
    else:
        dilation = 1
    layer4 = make_layer(in_channels, bottleneck, layer4_setting, width_mult, dilation)(layer3)
    layer5 = make_layer(in_channels, bottleneck, layer5_setting, width_mult, dilation)(layer4)

    # building last several layers
    last_in_channels = in_channels

    last_layer = ConvBNReLU_layer(last_in_channels, 1280, 1, initializer=initializer)(layer5)

    last_layer_shape = last_layer.shape.as_list()

    adv_avg = tf.keras.layers.AveragePooling2D(
        pool_size=(last_layer_shape[1], last_layer_shape[2]), strides=(last_layer_shape[1], last_layer_shape[2])
    )(last_layer)

    Dropout = tf.keras.layers.Dropout(rate=drop_rate)(adv_avg)
    conv2d = tf.keras.layers.Conv2D(filters=nclass, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(Dropout)

    output = tf.keras.layers.Flatten()(conv2d)

    return tf.keras.Model(inputs=input_layer, outputs=output,
                          name='frostnet_{}_{}_{}_{}'.format(
                              str(input_shape[0]) + 'x' + str(input_shape[1]) + 'x' + str(input_shape[2]),
                          nclass, mode, width_mult))


if __name__ == '__main__':
    input = np.random.randint(0, 10, size=(1, 224, 224, 3)).astype(np.float32)

    frostnet_model = FrostNet_inference(
        input_shape=(224, 224, 3),
        training=True,
        nclass=1000, mode='small', width_mult=1.0,
        bottleneck=CascadePreExBottleneck_layer)

    # output = frostnet_quant_large_1_25_model(input)
    frostnet_model.compile(optimizer='sgd')

    print(frostnet_model.summary())

    frostnet_model.save('./results/{}.h5'.format(frostnet_model.name))

    import os
    import sys

    cur_dir = os.path.dirname(sys.modules['__main__'].__file__)
    net_dir = os.path.split(cur_dir)[0]
    root = os.path.split(net_dir)[0]

    tf.keras.utils.plot_model(frostnet_model, to_file=root + '/results/{}.png'.format(frostnet_model.name),
                              show_shapes=True)
