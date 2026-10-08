import keras
from keras import Model, regularizers
from keras.src.applications.mobilenet_v3 import MobileNetV3Large
from keras.src.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input

from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.src.metrics import CategoricalCrossentropy

from keras.src.metrics.f_score_metrics import F1Score
from keras.src.optimizers import Adam


def create_mobile_net_v3_large(img_size: tuple):
    mobile_net = MobileNetV3Large(
        input_shape=(*img_size, 3),
        weights='imagenet',
        include_top=False
    )
    mobile_net.trainable = False
    return mobile_net, mobilenet_preprocess_input


def create_classification_model(deep_model,
                                num_classes: int,
                                reg_type: str=None,
                                is_dropout: bool=False,
                                lambdas=None,
                                learning_rate: float = 1e-3) -> Model:
    keras.utils.set_random_seed(42)
    def make_reg():
        if reg_type == "L1":
            return regularizers.l1(lambdas)
        if reg_type == "L2":
            return regularizers.l2(lambdas)
        if reg_type == "ElasticNet":
            l1, l2 = lambdas
            return regularizers.l1_l2(l1=l1, l2=l2)
        return None

    x = deep_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=make_reg())(x)
    if is_dropout:
        x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=make_reg())(x)
    if is_dropout:
        x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=deep_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score(average='macro'), CategoricalCrossentropy(name='ce')],
    )
    return model