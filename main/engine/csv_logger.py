"""
Minimal per-epoch CSV logger, kept separate from experience.log_dir
(./experiments_runs/<experiment_name>/ by default) so these CSVs can be
kept/diffed/shared independently of the full run directory (checkpoints,
Hydra's own logs, etc).

Two files per experiment, not one, to sidestep a header-consistency problem:
training losses are logged every epoch with a stable key set (the criterion
list doesn't change mid-run), while evaluation metrics only appear on eval
epochs but are also stable in their own right (evaluate() always returns the
same metric set) -- so each gets its own CSV with a header fixed at first
write, rather than one file needing a header that covers columns that don't
exist yet on epoch 1.
"""
import csv
import os


class EpochCSVLogger:
    def __init__(self, csv_dir, experiment_name, suffix):
        os.makedirs(csv_dir, exist_ok=True)
        self.path = os.path.join(csv_dir, f"{experiment_name}{suffix}.csv")
        self._fieldnames = None
        # Resume support: if the file already exists (e.g. experience.resume),
        # reuse its existing header instead of overwriting/duplicating it.
        if os.path.exists(self.path):
            with open(self.path, "r", newline="") as f:
                header = next(csv.reader(f), None)
                if header:
                    self._fieldnames = header

    def log(self, row):
        """row: dict of {column_name: value}. The header is fixed from the
        first row this logger ever writes (across the process's lifetime, or
        from the existing file on disk on resume). A later row with keys
        outside that header has them dropped (logged as a warning) rather
        than crashing a training run over a logging mismatch; a row missing
        some of the header's keys gets blanks for those columns.
        """
        write_header = self._fieldnames is None
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        unknown = set(row.keys()) - set(self._fieldnames)
        if unknown:
            import main.utils as lib
            lib.LOGGER.warning(
                f"EpochCSVLogger({self.path}): dropping columns not in this "
                f"CSV's header (fixed at first write): {sorted(unknown)}"
            )

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore", restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
