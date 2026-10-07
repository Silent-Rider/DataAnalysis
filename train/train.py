from keras import Model
from keras.src.callbacks import CSVLogger

# noinspection PyUnresolvedReferences
from train.log_callback import LogCallback


def fit_model(model: Model,
              train_dataset,
              val_dataset,
              epochs: int,
              model_name: str,
              initial_epoch: int = 0,
              logging: bool = False):
    callbacks = []
    if logging:
        text_logger = LogCallback(f'{model_name}.txt')
        csv_logger = CSVLogger(f'{LogCallback.LOG_FOLDER}/{model_name}.csv', append=False)
        callbacks.append(text_logger)
        callbacks.append(csv_logger)

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch)

    return history.history