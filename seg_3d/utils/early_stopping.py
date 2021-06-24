import logging


class EarlyStopping:
    def __init__(self, monitor, patience, mode):
        self.monitor = monitor
        self.patience = patience  # set to 0 to disable
        self.mode = mode
        assert mode in ["max", "min"], "\'mode\' must be either \'max\' or \'min\'"
        self.logger = logging.getLogger(__name__)
        self.fall_short_count = 0
        self.best = None
        self.triggered = False

    def check_is_valid(self, metric_list, class_labels="") -> None:
        metric_keys = ["val_loss", *metric_list.metrics.keys()]
        monitor = self.monitor.split("/")
        is_valid = True

        if len(monitor) > 1:
            metric_name, label_name = monitor
            if metric_name not in metric_keys or label_name not in class_labels:
                is_valid = False

        elif self.monitor not in metric_keys:
            is_valid = False

        if not is_valid:
            self.logger.warning("Early stopping enabled but cannot find metric: \'%s\'" % self.monitor)
            self.logger.warning("Options for monitored metrics are: [%s]" % ", ".join(map(str, metric_keys)))
            self.logger.warning("Disabling early stopping by setting patience to 0...")
            # disable early stopping
            self.patience = 0

    def get_es_result(self, current) -> bool:
        """Returns true if monitored metric has been improved"""
        if self.mode == 'max':
            return current > self.best
        elif self.mode == 'min':
            return current < self.best

    def check_early_stopping(self, metric_results) -> bool:
        if not self.patience:
            self.logger.warning("Early stopping disabled, skipping...")
            return False

        else:
            current = metric_results[self.monitor]

            if self.best is None:
                self.best = current

            elif self.get_es_result(current):
                self.best = current
                self.fall_short_count = 0
                self.logger.info("Best metric \'%s\' improved to %0.4f" % (self.monitor, current))
                return True

            else:
                self.fall_short_count += 1
                self.logger.info(
                    "Early stopping metric \'%s\' did not improve (count = %.0f), current %.04f, best %.04f" %
                    (self.monitor, self.fall_short_count, current, self.best))

        if self.fall_short_count >= self.patience:
            self.logger.info("Early stopping triggered, metric \'%s\' has not improved for %s validation steps" %
                             (self.monitor, self.patience))
            self.triggered = True

        return False
