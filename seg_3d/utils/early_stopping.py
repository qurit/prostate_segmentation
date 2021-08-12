import logging
from typing import List, Dict


class EarlyStopping:
    """
    Class for enabling early stopping during training
    """
    def __init__(self, monitor: str, patience: int, mode: str):
        """
        Args:
            monitor: name of the metric to monitor
            patience: number of steps before triggering early stopping
            mode: whether objective is to minimize of maximize metric
        """
        self.monitor = monitor
        self.patience = patience  # set to 0 to disable
        self.mode = mode
        assert mode in ["max", "min"], "\'mode\' must be either \'max\' or \'min\'"
        self.logger = logging.getLogger(__name__)
        self.fall_short_count = 0
        self.best = None
        self.triggered = False

    def check_is_valid(self, metrics: List[str], class_labels: List[str] = "") -> None:
        """
        Checks if the configuration is valid given the metric monitored and the metrics computed during eval

        Args:
            metrics: metric names computed during eval
            class_labels: labels for the different classes
        """
        metric_keys = ["val_loss", *metrics]  # val loss always gets computed during eval
        monitor = self.monitor.split("/")  # we assume metric names are of the form 'metric_name/label_name'
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

    def get_es_result(self, current: float) -> bool:
        """
        Checks if score has improved

        Args:
            current: score achieved at the current eval step

        Returns:
            True if current has improved over self.best
        """
        if self.mode == 'max':
            return current > self.best
        elif self.mode == 'min':
            return current < self.best

    def check_early_stopping(self, metric_results: Dict[str, float]) -> bool:
        """
        Updates early stopping state based on metric results

        Args:
            metric_results: dictionary containing the scores of all the metrics

        Returns:
            False if early stopping is disabled, patience has exceeded, or if metric has not improved
            True if metric has improved
        """
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
                    (self.monitor, self.fall_short_count, current, self.best)
                )

        if self.fall_short_count >= self.patience:
            self.logger.info("Early stopping triggered, metric \'%s\' has not improved for %s validation steps" %
                             (self.monitor, self.patience))
            self.triggered = True

        return False
