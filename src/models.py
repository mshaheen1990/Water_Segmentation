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


def se_block(x, ratio=8, name="se"):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    s = layers.Dense(max(ch // ratio, 1), activation="relu", name=f"{name}_fc1")(s)
    s = layers.Dense(ch, activation="sigmoid", name=f"{name}_fc2")(s)
    s = layers.Reshape((1, 1, ch), name=f"{name}_reshape")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])


def attention_gate(x, g, inter, name="att"):
    theta = layers.Conv2D(inter, 1, padding="same", name=f"{name}_theta")(x)
    phi = layers.Conv2D(inter, 1, padding="same", name=f"{name}_phi")(g)
    add = layers.Add(name=f"{name}_add")([theta, phi])
    act = layers.ReLU(name=f"{name}_relu")(add)
    psi = layers.Conv2D(1, 1, activation="sigmoid", name=f"{name}_psi")(act)
    return layers.Multiply(name=f"{name}_mul")([x, psi])


def build_unet_binary(input_shape=(64, 64, 9), base_filters=32, use_bn=True, dropout=0.0, variant="unet"):
    i = layers.Input(shape=input_shape, name="multiband_input")
    e1 = conv_block(i, base_filters, "enc1", use_bn, 0.0)
    p1 = layers.MaxPooling2D(2)(e1)
    e2 = conv_block(p1, base_filters * 2, "enc2", use_bn, 0.0)
    p2 = layers.MaxPooling2D(2)(e2)
    e3 = conv_block(p2, base_filters * 4, "enc3", use_bn, dropout)
    p3 = layers.MaxPooling2D(2)(e3)
    b = conv_block(p3, base_filters * 8, "bridge", use_bn, dropout)

    if variant == "se_unet":
        e1, e2, e3, b = se_block(e1, name="se1"), se_block(e2, name="se2"), se_block(e3, name="se3"), se_block(b, name="se4")

    u3 = layers.UpSampling2D(2)(b)
    s3 = attention_gate(e3, u3, base_filters * 2, name="att3") if variant == "attention_unet" else e3
    d3 = conv_block(layers.Concatenate()([u3, s3]), base_filters * 4, "dec3", use_bn, dropout)
    u2 = layers.UpSampling2D(2)(d3)
    s2 = attention_gate(e2, u2, base_filters, name="att2") if variant == "attention_unet" else e2
    d2 = conv_block(layers.Concatenate()([u2, s2]), base_filters * 2, "dec2", use_bn, 0.0)
    u1 = layers.UpSampling2D(2)(d2)
    s1 = attention_gate(e1, u1, max(base_filters // 2, 1), name="att1") if variant == "attention_unet" else e1

    if variant == "tfn_unet":
        t = layers.Conv2D(base_filters, 1, activation="relu", name="tfn_planet_proj")(i[..., :8])
        h = layers.Conv2D(base_filters, 1, activation="relu", name="tfn_hec_proj")(i[..., 8:9])
        fused = layers.Multiply(name="tfn_mul")([t, h])
        u1 = layers.Concatenate(name="tfn_concat")([u1, fused])

    d1 = conv_block(layers.Concatenate()([u1, s1]), base_filters, "dec1", use_bn, 0.0)
    o = layers.Conv2D(1, 1, activation="sigmoid", name="water_mask")(d1)
    return Model(i, o, name=f"{variant}_early_fusion")
