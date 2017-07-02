import os
from time import localtime, strftime


class ProjectPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, logdir):
        self.logdir = logdir

        self.timestamp = strftime("%B_%d__%H:%M", localtime())
        self.model_path = os.path.join(ProjectPath.base, self.logdir, self.timestamp)
        self.data_path = os.path.join(ProjectPath.base, "data")
