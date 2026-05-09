from tensorflow.keras import Model, layers


def conv_block(x, filters, name, use_bn=True, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=not use_bn, name=f"{name}_conv1")(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=not use_bn, name=f"{name}_conv2")(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def build_unet_binary(input_shape=(64, 64, 9), base_filters=32, use_bn=True, dropout=0.0):
    i = layers.Input(shape=input_shape, name="multiband_input")
    e1 = conv_block(i, base_filters, "enc1", use_bn, 0.0)
    p1 = layers.MaxPooling2D(2)(e1)
    e2 = conv_block(p1, base_filters * 2, "enc2", use_bn, 0.0)
    p2 = layers.MaxPooling2D(2)(e2)
    e3 = conv_block(p2, base_filters * 4, "enc3", use_bn, dropout)
    p3 = layers.MaxPooling2D(2)(e3)
    b = conv_block(p3, base_filters * 8, "bridge", use_bn, dropout)
    u3 = layers.UpSampling2D(2)(b)
    d3 = conv_block(layers.Concatenate()([u3, e3]), base_filters * 4, "dec3", use_bn, dropout)
    u2 = layers.UpSampling2D(2)(d3)
    d2 = conv_block(layers.Concatenate()([u2, e2]), base_filters * 2, "dec2", use_bn, 0.0)
    u1 = layers.UpSampling2D(2)(d2)
    d1 = conv_block(layers.Concatenate()([u1, e1]), base_filters, "dec1", use_bn, 0.0)
    o = layers.Conv2D(1, 1, activation="sigmoid", name="water_mask")(d1)
    return Model(i, o, name="unet_early_fusion")
