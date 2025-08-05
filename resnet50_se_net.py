from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, GlobalAveragePooling2D, Reshape, Dense, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K

def squeeze_excite_block(input, ratio=16):
    """Create a squeeze and excitation block."""
    init = input
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def conv_block(input, num_filters):
    """Convolution Block with SE Block."""
    x = Conv2D(num_filters, (3, 3), padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Adding SE Block after convolutions
    x = squeeze_excite_block(x)

    return x

def decoder_block(input, skip_features, num_filters):
    """Decoder Block."""
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50_se_net(input_shape):
    """Builds the ResNet50-Squeeze and Excitation-Net."""
    inputs = Input(input_shape)

    # Pre-trained ResNet50 Model
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Encoder with SE Blocks
    s1 = resnet50.get_layer("input_1").output
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output

    # Bridge
    b1 = resnet50.get_layer("conv4_block6_out").output

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="ResNet50-SE-Net")
    return model

if __name__ == "__main__":
    input_shape = (576, 576, 3)
    model = build_resnet50_se_net(input_shape)
    model.summary()
