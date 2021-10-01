from collections import defaultdict

import pandas as pd


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)
