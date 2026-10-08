import time
from tensorflow.keras.callbacks import Callback


class LogCallback(Callback):
    LOG_FOLDER = f'artifacts/logs'
    train_start_time = 0

    def __init__(self, file_name='model_log.txt'):
        super().__init__()
        self.file_name = file_name


    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()


    def on_epoch_end(self, epoch, logs=None):
        elapsed_total = time.time() - self.train_start_time

        acc = logs.get('accuracy', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)

        f1 = logs.get('f1_score')
        val_f1 = logs.get('val_f1_score')

        f1_metric = (f"f1_score: {f1:.4f} - val_f1_score: {val_f1:.4f}\n"
                             if f1 is not None else "")

        metrics = f"accuracy: {acc:.4f} - val_accuracy: {val_acc:.4f}\n{f1_metric}"

        ce = logs.get('ce', 0.0)
        val_ce = logs.get('val_ce', 0.0)
        metrics += f"ce: {ce:.4f} - val_ce: {val_ce:.4f}\n"

        log_msg = (
            f"\tEpoch {epoch + 1}/{self.params['epochs']}\n"
            f'{metrics}'
            f"total time: {int(elapsed_total // 60)} minutes {elapsed_total % 60:.2f} seconds\n\n"
        )

        with open(f'{self.LOG_FOLDER}/{self.file_name}', mode='a') as log_file:
            log_file.write(log_msg)
