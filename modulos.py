import tflearn
from tflearn.layers.conv import conv_2d
import tensorflow as tf

def rb_reference(incoming, nb_blocks, out_channels, downsample=False,
                 downsample_strides=2, activation='relu', batch_norm=True,
                 bias=True, weights_init='variance_scaling',
                 bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                 trainable=True, restore=True, reuse=False, scope=None,
                 name="ResidualBlock", bn_position='before', last_actv=True):
    """ Residual Block.

    A residual block as described in MSRA's Deep Residual Network paper.
    Full pre-activation architecture is used here.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            convolution layers.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ShallowBottleneck'.

    References:
        - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
        - Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.

    Links:
        - [http://arxiv.org/pdf/1512.03385v1.pdf]
            (http://arxiv.org/pdf/1512.03385v1.pdf)
        - [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027v2.pdf)

    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                                reuse=reuse) as scope:

        name = scope.name  # TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            resnet = conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels) // 2
                identity = tf.pad(identity,
                                       [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            if batch_norm and bn_position=='before':
                resnet = tflearn.batch_normalization(resnet)

            resnet = resnet + identity

            # La ultima activacion se hace despues de la identidad segun el paper
            if batch_norm and bn_position=='after':
                resnet = tflearn.batch_normalization(resnet)

            if last_actv:
                resnet = tflearn.activation(resnet, activation)

    return resnet

def rb_preactv(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock"):
    """ Residual Block.

    A residual block as described in MSRA's Deep Residual Network paper.
    Full pre-activation architecture is used here.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            convolution layers.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ShallowBottleneck'.

    References:
        - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
        - Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.

    Links:
        - [http://arxiv.org/pdf/1512.03385v1.pdf]
            (http://arxiv.org/pdf/1512.03385v1.pdf)
        - [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027v2.pdf)

    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            resnet = tflearn.activation(resnet, activation)

            resnet = tflearn.conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = tflearn.conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            resnet = resnet + identity

    return resnet

rb_full_preactv = tflearn.residual_block

def my_plain_block(incoming, nb_blocks, out_channels, downsample=False,
                      downsample_strides=2, activation='relu', batch_norm=True,
                      bias=True, weights_init='variance_scaling',
                      bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                      trainable=True, restore=True, reuse=False, scope=None,
                      name="ResidualBlock",bn_position='before',last_actv=True):
    """ Residual Block.

    A residual block as described in MSRA's Deep Residual Network paper.
    Full pre-activation architecture is used here.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            convolution layers.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ShallowBottleneck'.

    References:
        - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
        - Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.

    Links:
        - [http://arxiv.org/pdf/1512.03385v1.pdf]
            (http://arxiv.org/pdf/1512.03385v1.pdf)
        - [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027v2.pdf)

    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                                reuse=reuse) as scope:

        name = scope.name  # TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            resnet = conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels) // 2
                identity = tf.pad(identity,
                                       [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            if batch_norm and bn_position=='before':
                resnet = tflearn.batch_normalization(resnet)

            # En el bloque plano la identidad no se suma
            #resnet = resnet + identity

            # La ultima activacion se hace despues de la identidad segun el paper
            if batch_norm and bn_position=='after':
                resnet = tflearn.batch_normalization(resnet)

            if last_actv:
                resnet = tflearn.activation(resnet, activation)

    return resnet


def rb_sub(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock"):
    """ Residual Block.

    A residual block as described in MSRA's Deep Residual Network paper.
    Full pre-activation architecture is used here.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            convolution layers.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ShallowBottleneck'.

    References:
        - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
        - Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.

    Links:
        - [http://arxiv.org/pdf/1512.03385v1.pdf]
            (http://arxiv.org/pdf/1512.03385v1.pdf)
        - [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027v2.pdf)

    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels


            resnet = tf.subtract(identity,resnet)

    return resnet
