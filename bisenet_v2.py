# ------------------------------------------------------------------------------
# Written by Hamid Ali (hamidriasat@gmail.com)
# ------------------------------------------------------------------------------
import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.activations as activation


def stem_block(x_in, channels):
    x = layers.Conv2D(filters=channels, kernel_size=(3, 3), strides=2, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x_split = activation.relu(x)

    x = layers.Conv2D(filters=channels // 2, kernel_size=(1, 1), padding='same')(x_split)
    x = layers.BatchNormalization()(x)
    x = activation.relu(x)

    x = layers.Conv2D(filters=channels, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation.relu(x)

    y = layers.MaxPooling2D()(x_split)

    x = layers.Concatenate()([x, y])
    x = layers.Conv2D(filters=channels, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation.relu(x)

    return x


def context_embedding_block(x_in, channels):
    # To make it compatible with tf lite use AveragePooling2D instead of GlobalAveragePooling2D
    # x = layers.GlobalAveragePooling2D()(x_in)
    # x = layers.BatchNormalization()(x)
    # x = layers.Reshape((1, 1, c))(x)
    h4 = tf.keras.backend.int_shape(x_in)[1]
    w4 = tf.keras.backend.int_shape(x_in)[2]
    x = tf.keras.layers.AveragePooling2D(pool_size=(h4, w4), strides=(h4, w4))(x_in)

    x = layers.Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation.relu(x)

    x = layers.Add()([x, x_in])
    x = layers.Conv2D(filters=channels, kernel_size=(3, 3), padding='same')(x)
    return x


def gather_and_expansion_layer(x_in, channels, e=6, stride=1):
    """Gather And Expansion Layer implementation with both stride 1 and stride 2"""
    x = layers.Conv2D(filters=channels, kernel_size=(3, 3), padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = activation.relu(x)

    if stride == 2:
        x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)

        y = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), strides=2, padding='same')(x_in)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)
    else:
        y = x_in

    x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, y])
    x = activation.relu(x)

    return x


def conv_bn_relu(x_in, channels, stride=1):
    """Apply Conv2D, Batch Norm and Relu in sequential Order """
    x = layers.Conv2D(filters=channels, kernel_size=(3, 3), strides=stride, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = activation.relu(x)

    return x


def detail_branch(x_in, channels):
    """The detail branch of BiSeNet, which has wide channels but shallow layers."""
    c1, c2, c3 = channels

    # S1
    x = conv_bn_relu(x_in, c1, stride=2)
    x = conv_bn_relu(x, c1, stride=1)

    # S2
    x = conv_bn_relu(x, c2, stride=2)
    x = conv_bn_relu(x, c2, stride=1)
    x = conv_bn_relu(x, c2, stride=1)

    # S3
    x = conv_bn_relu(x, c3, stride=2)
    x = conv_bn_relu(x, c3, stride=1)
    x = conv_bn_relu(x, c3, stride=1)

    return x


def semantic_branch(x_in, channels):
    """The semantic branch of BiSeNet, which has narrow channels but deep layers."""

    channel_1, channel_c3, channel_c4, channel_c5 = channels

    stage2 = stem_block(x_in, channel_1)

    # S3
    stage3 = gather_and_expansion_layer(stage2, channel_c3, stride=2)
    stage3 = gather_and_expansion_layer(stage3, channel_c3, stride=1)

    # S4
    stage4 = gather_and_expansion_layer(stage3, channel_c4, stride=2)
    stage4 = gather_and_expansion_layer(stage4, channel_c4, stride=1)

    # S5
    stage5_4 = gather_and_expansion_layer(stage4, channel_c5, stride=2)
    stage5_4 = gather_and_expansion_layer(stage5_4, channel_c5, stride=1)
    stage5_4 = gather_and_expansion_layer(stage5_4, channel_c5, stride=1)
    stage5_4 = gather_and_expansion_layer(stage5_4, channel_c5, stride=1)

    fm = context_embedding_block(stage5_4, channel_c5)

    return stage2, stage3, stage4, stage5_4, fm


def bilateral_guided_aggregation_layer(detail, semantic, channels):
    """The Bilateral Guided Aggregation Layer, used to fuse the semantic features and spatial features."""

    # detail branch
    detail_a = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(detail)
    detail_a = layers.BatchNormalization()(detail_a)

    detail_a = layers.Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(detail_a)

    detail_b = layers.Conv2D(filters=channels, kernel_size=(3, 3), strides=2, padding='same')(detail)
    detail_b = layers.BatchNormalization()(detail_b)

    detail_b = layers.AveragePooling2D((3, 3), strides=2, padding='same')(detail_b)

    # semantic branch
    semantic_a = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(semantic)
    semantic_a = layers.BatchNormalization()(semantic_a)

    semantic_a = layers.Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(semantic_a)
    semantic_a = activation.sigmoid(semantic_a)

    semantic_b = layers.Conv2D(filters=channels, kernel_size=(3, 3), padding='same')(semantic)
    semantic_b = layers.BatchNormalization()(semantic_b)

    h2 = tf.keras.backend.int_shape(semantic_b)[1] * 4
    w2 = tf.keras.backend.int_shape(semantic_b)[2] * 4
    semantic_b = tf.image.resize(semantic_b, [h2, w2])
    semantic_b = activation.sigmoid(semantic_b)

    # combining
    detail = tf.multiply(detail_a, semantic_b)
    semantic = tf.multiply(semantic_a, detail_b)

    h3 = tf.keras.backend.int_shape(semantic)[1] * 4
    w3 = tf.keras.backend.int_shape(semantic)[2] * 4
    semantic = tf.image.resize(semantic, [h3, w3])

    x = layers.Add()([detail, semantic])
    x = layers.Conv2D(filters=channels, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    return x


def segmentation_head(x_in, mid_dim, num_classes):
    x = layers.Conv2D(filters=mid_dim, kernel_size=(3, 3), padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = activation.relu(x)

    x = layers.Dropout(.1)(x)

    x = layers.Conv2D(filters=num_classes, kernel_size=(1, 1), padding='same')(x)

    return x


def bisenet_v2(input_shape, num_classes=2, _lambda=0.25, training=False, from_logits=True):
    """
        The BiSeNet V2 implementation based on Tensorflow/Keras.
        The original article refers to
        Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
        (https://arxiv.org/abs/2004.02147)
        Args:
            input_shape (list(int)): Input shape list/tuple with shape height, width and channels
            num_classes (int): The unique number of target classes.
            _lambda (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
            training (bool, optional): Training mode. If false means it's eval mode, where auxiliary outputs are removed.
            from_logits (bool, optional): Whether to apply activation function on finial layer or not
        """

    x_in = layers.Input(input_shape)  # input layer

    # create channels size for all branches
    C1, C2, C3 = 64, 64, 128
    db_channels = (C1, C2, C3)
    C1, C3, C4, C5 = int(C1 * _lambda), int(C3 * _lambda), 64, 128
    sb_channels = (C1, C3, C4, C5)
    mid_channels = 128

    #  detail branch
    db_out = detail_branch(x_in, db_channels)

    # semantic branch
    feat1, feat2, feat3, feat4, sfm = semantic_branch(x_in, sb_channels)

    logit = bilateral_guided_aggregation_layer(db_out, sfm, mid_channels)

    logit = segmentation_head(logit, mid_channels, num_classes)

    if not training:
        output_list = [logit]
    else:
        logit_1 = segmentation_head(feat1, C1, num_classes)
        logit_2 = segmentation_head(feat2, C3, num_classes)
        logit_3 = segmentation_head(feat3, C4, num_classes)
        logit_4 = segmentation_head(feat4, C5, num_classes)
        output_list = [logit, logit_1, logit_2, logit_3, logit_4]

    output_list = [tf.image.resize(logit, input_shape[:2]) for logit in output_list]

    if not from_logits and num_classes == 2:
        output_list = [tf.nn.sigmoid(logit) for logit in output_list]
    elif not from_logits:
        output_list = [tf.nn.softmax(logit) for logit in output_list]

    model = models.Model(inputs=[x_in], outputs=output_list)

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model


if __name__ == "__main__":
    """## Model Compilation"""
    # default input shape
    INPUT_SHAPE = (512, 1024, 3)
    OUTPUT_CHANNELS = 19
    with tf.device("cpu:0"):
        bisenet_v2_model = bisenet_v2(
            input_shape=INPUT_SHAPE,
            num_classes=OUTPUT_CHANNELS
        )
        optimizer = optimizers.SGD(
            momentum=0.9,
            lr=0.001
        )
        bisenet_v2_model.compile(
            loss=losses.CategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=['accuracy']
        )
        bisenet_v2_model.summary()

        # tf.keras.utils.plot_model(bisenet_v2_model, show_layer_names=True, show_shapes=True)
        # bisenet_v2_model.save("./bisenet_v2_model.hdf5")
        print("Done")
