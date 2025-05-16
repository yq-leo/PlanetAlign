import os
import csv
import json
import datetime
from typing import List, Dict, Optional, Union, Tuple


class TrainingLogger:
    """Logger class for tracking the training process of a network alignment algorithm.

    Parameters
    ----------
    log_dir : str
        Directory where logs are stored.
    log_name : str, optional
        Custom log file name. Defaults to timestamp-based naming.
    top_ks : List[int], optional
        List of K values for Hits@K metrics. Defaults to [1, 10, 30, 50].
    digits : int, optional
        Number of decimal places for metrics. Defaults to 4.
    additional_headers : List[str], optional
        Additional headers for the log file. Defaults to None.
    save : bool, optional
        Flag to save logs to file. Defaults to True.
    """
    def __init__(self,
                 log_dir: str = "logs",
                 log_name: Optional[str] = None,
                 top_ks: Union[List[int], Tuple[int, ...]] = (1, 10, 30, 50),
                 digits: int = 4,
                 additional_headers: Optional[List[str]] = None,
                 save: bool = True):

        self.top_ks = top_ks
        self.digits = digits
        headers = ["Epoch", "Loss", "EpochTime"] + [f"Hits@{k}" for k in top_ks] + ["MRR"]
        if additional_headers:
            headers += additional_headers
        self.headers = headers
        self.logs = []  # Stores training history in-memory

        self.save = save
        if self.save:
            os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
            if log_name is None:
                log_name = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.log_path = os.path.join(log_dir, log_name)

            # Initialize log file
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)  # Write column headers

    def log(self,
            epoch: int,
            loss: float,
            epoch_time: float,
            hits: Dict[int, float],
            mrr: float,
            verbose: bool = True,
            **kwargs):
        """
        Logs a single training step.

        Parameters:
        -----------
        epoch : int
            Current training epoch.
        loss : float
            Training loss value.
        time: float
            Time taken for the current epoch.
        hits : Dict[int, float]
            Dictionary of Hits@K values (e.g., {1: 0.5, 10: 0.7, 30: 0.8, 50: 0.85}).
        mrr : float
            Mean Reciprocal Rank (MRR) score.
        verbose : bool
            Flag for printing the log to console.
        """
        assert all([f'Hits@{k}' in self.headers for k in hits.keys()]), "Invalid keys in hits dictionary"
        assert all([key in self.headers for key in kwargs.keys()]), "Invalid keys in additional parameters"

        log_entry = {
            "Epoch": epoch,
            "Loss": loss,
            "EpochTime": round(epoch_time, self.digits),
            "MRR": round(mrr, self.digits)
        }
        log_entry.update({f"Hits@{k}": round(v, self.digits) for k, v in hits.items()})
        log_entry.update(kwargs)
        self.logs.append(log_entry)

        # Save to CSV
        if self.save:
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(log_entry)

        # Print log to console
        if verbose:
            print(self.format_log(log_entry))

    def log_comments(self, comments: Union[str, List[str], Tuple[str, ...]]):
        """
        Logs comments.

        Parameters:
        -----------
        comments : List[str]
            List of comments to log.
        """

        if isinstance(comments, str):
            comments = [comments]
        for comment in comments:
            print(comment)

        if self.save:
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for comment in comments:
                    writer.writerow(["# " + comment])

    @staticmethod
    def format_log(log_entry: Dict) -> str:
        """
        Formats the log entry for printing.
        """

        title = f" Epoch {log_entry['Epoch']:03d} | Loss: {log_entry['Loss']:.6f} | EpochTime: {log_entry['EpochTime']:.2f}s "
        metrics_line = " | ".join([f"{key}: {val:.4f}" for key, val in log_entry.items() if key.startswith("Hits@")]) + \
                       f" | MRR: {log_entry['MRR']:.4f}"
        cols = ['Epoch', 'Loss', 'EpochTime', 'MRR']
        add_info = " | ".join([f"{key}: {val}" for key, val in log_entry.items() if not key.startswith("Hits@") and key not in cols])

        line_length = max(len(title), len(metrics_line), len(add_info))
        title_str = title.center(line_length, '-')
        metrics_str = metrics_line.center(line_length, '-')
        add_info_str = add_info.center(line_length, '-')

        return f"{title_str}\n{metrics_str}\n{add_info_str}"

    def save_json(self):
        """
        Saves the logs as a JSON file for structured analysis.
        """
        json_path = self.log_path.replace(".csv", ".json")
        with open(json_path, "w") as f:
            json.dump(self.logs, f, indent=4)
        print(f"Logs saved to {json_path}")

    def summary(self):
        """
        Prints a summary of the training process.
        """
        if not self.logs:
            print("No logs recorded yet.")
            return

        best_mrr = max(self.logs, key=lambda x: x["MRR"])
        print("=" * 60)
        print(f"Training Summary - Best MRR at Epoch {best_mrr['Epoch']}: {best_mrr['MRR']:.4f}")
        print("=" * 60)

    def get_logs(self) -> List[Dict]:
        """
        Returns the logs as a list of dictionaries.
        """
        return self.logs
