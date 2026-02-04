import logging
import csv
import os
from datetime import datetime
from lightning.pytorch.loggers import CSVLogger


class NebulaCSVLogger(CSVLogger):
    def __init__(self, scenario_start_time, save_dir, name="metrics", version=None, *args, **kwargs):
        self.scenario_start_time = scenario_start_time
        # We perform our own step tracking
        self.local_step = 0
        self.global_step = 0
        self.current_round = 0
        
        # Initialize parent with typical args
        super().__init__(save_dir, name, version, *args, **kwargs)
        
        # Ensure log directory exists immediately
        self._custom_log_dir = self.log_dir
        os.makedirs(self._custom_log_dir, exist_ok=True)

    @property
    def log_dir(self):
        return super().log_dir

    def get_step(self):
        try:
            return int((datetime.now() - datetime.strptime(self.scenario_start_time, "%d/%m/%Y %H:%M:%S")).total_seconds())
        except:
            return 0

    def log_data(self, data, step=None):
        if step is None:
            step = self.get_step()
        
        # Forward to log_metrics which handles the dispatching
        try:
            self.log_metrics(data, step)
        except Exception as e:
            logging.exception(f"Error logging data [{data}] for step [{step}]: {e}")

    def log_metrics(self, metrics, step=None):
        if step is None:
            self.local_step += 1
            step = self.global_step + self.local_step

        if "epoch" in metrics:
            metrics.pop("epoch")
            
        # 1. Check for Round Update
        if "A-Round" in metrics:
            self.current_round = metrics["A-Round"]

        # 2. Classify Metrics
        resource_metrics = {}
        model_metrics = {}
        
        for k, v in metrics.items():
            # Resource prefixes based on reporter.py: W-CPU, Z-RAM, Y-Disk, X-Network
            if k.startswith(("W-", "X-", "Y-", "Z-")):
                resource_metrics[k] = v
            else:
                model_metrics[k] = v

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3. Write Resource Metrics to resources.csv
        if resource_metrics:
            self._append_to_csv("resources.csv", resource_metrics, step, timestamp)

        # 4. Write Model Metrics to specific files
        if model_metrics:
            # Determine phase from keys (e.g., Train/Loss, Validation/Accuracy)
            phase = "general"
            for k in model_metrics.keys():
                if "Train/" in k:
                    phase = "train"
                    break
                elif "Validation/" in k:
                    phase = "validation"
                    break
                elif "Test (Local)/" in k:
                    phase = "test_local"
                    break
                elif "Test (Global)/" in k:
                    phase = "test_global"
                    break
            
            filename = f"metrics_{phase}_round_{self.current_round}.csv"
            self._append_to_csv(filename, model_metrics, step, timestamp)

    def _append_to_csv(self, filename, metrics_dict, step, timestamp):
        filepath = os.path.join(self.log_dir, filename)
        file_exists = os.path.isfile(filepath)
        
        # Prepare row data
        row_data = {"step": step, "timestamp": timestamp}
        # Clean values (tensor -> float if needed)
        for k, v in metrics_dict.items():
            try:
                row_data[k] = float(v) if hasattr(v, 'item') else v
            except:
                row_data[k] = v

        fieldnames = ["step", "timestamp"] + sorted(metrics_dict.keys())
        
        try:
            if not file_exists:
                with open(filepath, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row_data)
            else:
                with open(filepath, 'r') as r:
                    reader = csv.reader(r)
                    try:
                        existing_header = next(reader)
                    except StopIteration:
                        existing_header = []

                # Identify new columns
                new_cols = [col for col in fieldnames if col not in existing_header]

                if new_cols:
                    rows = []
                    with open(filepath, 'r') as r:
                        reader = csv.DictReader(r)
                        rows = list(reader)
                    
                    final_fieldnames = existing_header + new_cols
                    
                    with open(filepath, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                        writer.writerow(row_data)
                else:
                    with open(filepath, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=existing_header, extrasaction='ignore')
                        writer.writerow(row_data)

        except Exception as e:
            logging.warning(f"Failed to write to CSV {filepath}: {e}")

    def log_figure(self, figure, step=None, name=None):
        if step is None:
            step = self.get_step()
        try:
            # Save figure to the log directory
            safe_name = str(name).replace("/", "_") if name else "figure"
            os.makedirs(self.log_dir, exist_ok=True)
            save_path = f"{self.log_dir}/{safe_name}_step_{step}.png"
            figure.savefig(save_path)
        except Exception as e:
            logging.warning(f"Error saving figure [{name}] to CSV logger: {e}")

    def get_logger_config(self):
        return {
            "scenario_start_time": self.scenario_start_time,
            "local_step": self.local_step,
            "global_step": self.global_step,
            "current_round": self.current_round
        }

    def set_logger_config(self, logger_config):
        if logger_config is None:
            return
        try:
            self.scenario_start_time = logger_config["scenario_start_time"]
            self.local_step = logger_config["local_step"]
            self.global_step = logger_config["global_step"]
            self.current_round = logger_config.get("current_round", 0)
        except Exception as e:
            logging.exception(f"Error setting logger config: {e}")
