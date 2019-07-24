from visdom import Visdom
import numpy as np


class Plot():
    def __init__(self,win):
        self.viz = Visdom(env=win)
        self.name = {}
        self.loss_windows = {}
        self.epoch = {}

    def plot(self, name=None,epoch=None):
        self.epoch=epoch
        for epoch_name,val in self.epoch.items():
            epo=val
        for i, axis_name in enumerate(name.keys()):
            if axis_name not in self.name:
                self.name[axis_name] = name[axis_name]
            else:
                self.name[axis_name] += name[axis_name]
        for axis_name, value in self.name.items():
                if axis_name not in self.loss_windows:
                    self.loss_windows[axis_name] = self.viz.line(X=np.array([epo]),
                                                                 Y=np.array([value]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': axis_name,
                                                                       'title': axis_name})

                else:
                    self.viz.line(X=np.array([epo]), Y=np.array([value]),
                                  win=self.loss_windows[axis_name], update='append')
                self.name[axis_name] = 0.0