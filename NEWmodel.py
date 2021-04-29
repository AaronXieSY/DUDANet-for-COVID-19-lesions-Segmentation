from keras.layers import *
from keras.models import *


IMAGE_ORDERING = 'channels_last'


def conv_bn_relu(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1
    x = Conv2D(filters, kernel, strides=strides, padding="same",
               data_format=IMAGE_ORDERING)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation('relu')(x)


def reduce_mean(inputs):
    return K.mean(inputs, axis=(1, 2), keepdims=True)


def CH_attention(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1
    f = Conv2D(filters, kernel, strides=strides, padding="same",
                data_format=IMAGE_ORDERING, use_bias=False)(inputs)
    f = BatchNormalization(axis=channel_axis)(f)
    f = Activation('relu')(f)
    f1 = Conv2D(filters, kernel, strides=strides, padding="same",
                data_format=IMAGE_ORDERING, dilation_rate=(2, 2), use_bias=False)(f)
    f1 = BatchNormalization(axis=channel_axis)(f1)
    f1 = Activation('relu')(f1)
    f1 = Lambda(reduce_mean)(f1)
    f1 = Conv2D(filters, (1, 1), activation='sigmoid')(f1)
    f2 = Conv2D(filters, kernel, strides=strides, padding="same",
                data_format=IMAGE_ORDERING, dilation_rate=(4, 4), use_bias=False)(f)
    f2 = BatchNormalization(axis=channel_axis)(f2)
    f2 = Activation('relu')(f2)
    f2 = Lambda(reduce_mean)(f2)
    f2 = Conv2D(filters, (1, 1), activation='sigmoid')(f2)
    f3 = Conv2D(filters, kernel, strides=strides, padding="same",
                data_format=IMAGE_ORDERING, dilation_rate=(6, 6), use_bias=False)(f)
    f3 = BatchNormalization(axis=channel_axis)(f3)
    f3 = Activation('relu')(f3)
    f3 = Lambda(reduce_mean)(f3)
    f3 = Conv2D(filters, (1, 1), activation='sigmoid')(f3)
    f4 = Conv2D(filters, kernel, strides=strides, padding="same",
                data_format=IMAGE_ORDERING, dilation_rate=(8, 8), use_bias=False)(f)
    f4 = BatchNormalization(axis=channel_axis)(f4)
    f4 = Activation('relu')(f4)
    f4 = Lambda(reduce_mean)(f4)
    f4 = Conv2D(filters, (1, 1), activation='sigmoid')(f4)
    x = Concatenate(axis=-1)([f1, f2, f3, f4])
    outdim = x.get_shape().as_list()[-1]
    full1 = Dense(int(outdim / 4), use_bias=True)(x)
    full1 = Activation('relu')(full1)
    full2 = Dense(filters, use_bias=True)(full1)
    full2 = Activation('sigmoid')(full2)
    xx = multiply([f, full2])
    xxx = add([f, xx])
    return xxx

## 粗分割网络
def Net1(input_height=256, input_weight=256):

    input = Input(shape=(input_height, input_weight, 1))

    conv1 = conv_bn_relu(input, 32, kernel=(3, 3), strides=(1, 1))
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv_bn_relu(pool1, 64, kernel=(3, 3), strides=(1, 1))
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv_bn_relu(pool2, 128, kernel=(3, 3), strides=(1, 1))
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv_bn_relu(pool3, 256, kernel=(3, 3), strides=(1, 1))
    pool4 = MaxPooling2D((2, 2))(conv4)

    deconv4 = conv_bn_relu(pool4, 512, kernel=(1, 1), strides=(1, 1))
    deconv4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(deconv4)
    uconv4 = concatenate([deconv4, conv4])

    deconv3 = conv_bn_relu(uconv4, 256, kernel=(1, 1), strides=(1, 1))
    deconv3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(deconv3)
    uconv3 = concatenate([deconv3, conv3])

    deconv2 = conv_bn_relu(uconv3, 128, kernel=(1, 1), strides=(1, 1))
    deconv2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(deconv2)
    uconv2 = concatenate([deconv2, conv2])

    deconv1 = conv_bn_relu(uconv2, 64, kernel=(1, 1), strides=(1, 1))
    deconv1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(deconv1)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = conv_bn_relu(uconv1, 32, kernel=(1, 1), strides=(1, 1))

    output_layer = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING,
                          activation='sigmoid', name='output_layer')(uconv1)  # 1*1*1
    model = Model(inputs=input, outputs=output_layer)
    model.model_name = 'Net1'
    return model

## 细分割网络
def Net2(input_height=256, input_weight=256):

    input = Input(shape=(input_height, input_weight, 1))

    conv1 = conv_bn_relu(input, 32, kernel=(3, 3), strides=(1, 1))
    conv1 = CH_attention(conv1, 64)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv_bn_relu(pool1, 64, kernel=(3, 3), strides=(1, 1))
    conv2 = CH_attention(conv2, 128)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv_bn_relu(pool2, 128, kernel=(3, 3), strides=(1, 1))
    conv3 = CH_attention(conv3, 256)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv_bn_relu(pool3, 256, kernel=(3, 3), strides=(1, 1))
    conv4 = CH_attention(conv4, 512)
    pool4 = MaxPooling2D((2, 2))(conv4)

    deconv4 = conv_bn_relu(pool4, 512, kernel=(1, 1), strides=(1, 1))
    deconv4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(deconv4)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = CH_attention(uconv4, 256)

    deconv3 = conv_bn_relu(uconv4, 256, kernel=(1, 1), strides=(1, 1))
    deconv3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(deconv3)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = CH_attention(uconv3, 128)

    deconv2 = conv_bn_relu(uconv3, 128, kernel=(1, 1), strides=(1, 1))
    deconv2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = CH_attention(uconv2, 64)

    deconv1 = conv_bn_relu(uconv2, 64, kernel=(1, 1), strides=(1, 1))
    deconv1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(deconv1)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = CH_attention(uconv1, 32)

    uconv1 = conv_bn_relu(uconv1, 32, kernel=(1, 1), strides=(1, 1))

    output_layer = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING,
                          activation='sigmoid', name='output_layer')(uconv1)  # 1*1*1
    model = Model(inputs=input, outputs=output_layer)
    model.model_name = 'Net2'
    return model



