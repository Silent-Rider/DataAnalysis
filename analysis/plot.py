import matplotlib.pyplot as plt

METRICS_MAP = {
    'ce': ('кросс-энтропии', 'Кросс-энтропия'),
    'accuracy': ('точности (Accuracy)', 'Точность'),
    'f1_score': ('F1-меры', 'F1-мера')
}


def draw_subplots(history:dict, filename:str=None, title:str=None):
    metrics: list[tuple[str, tuple]] = []
    for metric in METRICS_MAP.keys():
        train_values = history.get(metric)
        val_values = history.get(f"val_{metric}")
        if train_values and val_values:
            metrics.append((metric, (train_values, val_values)))
    if len(metrics) == 0:
        return

    plt.figure(figsize=(15, 5))
    if title:
        plt.suptitle(title)
    for i, (metric_name, metric_values) in enumerate(metrics):
        metric_title, metric_ylabel = METRICS_MAP[metric_name]
        plt.subplot(1, len(metrics), i + 1)
        prepare_plot_colored(*metric_values, metric_title, metric_ylabel)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if filename:
        plt.savefig(f'artifacts/plots/{filename}', dpi=300, bbox_inches='tight')
    plt.show()


def prepare_plot_colored(train_data, val_data, title:str, ylabel:str):
    epochs_iter = list(range(1, len(train_data) + 1))
    plt.plot(epochs_iter, train_data, label='Обучающая выборка', marker='o', color='tab:blue')
    plt.plot(epochs_iter, val_data, label='Валидационная выборка', marker='s', color='tab:orange')
    plt.title('Динамика ' + title)
    plt.xlabel('Эпоха')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)