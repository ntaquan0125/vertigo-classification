import pyqtgraph as pg

from PyQt5 import QtWidgets, uic


class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('vng_analyzer.ui', self)
        self.show()

        self.horizontal_plot.getPlotItem().setTitle(
            '<span style="color:red">Horizontal right</span> <span style="color:blue">Horizontal left</span>'
        )
        self.horizontal_plot.getPlotItem().showGrid(x=True, y=True)
        self.vertical_plot.getPlotItem().setTitle(
            '<span style="color:red">Vertical right</span> <span style="color:blue">Vertical left</span>'
        )
        self.vertical_plot.getPlotItem().showGrid(x=True, y=True)

        self.xr_curve = self.horizontal_plot.plot(pen=pg.mkPen('r', width=1.5))
        self.xl_curve = self.horizontal_plot.plot(pen=pg.mkPen('b', width=1.5))
        self.yr_curve = self.vertical_plot.plot(pen=pg.mkPen('r', width=1.5))
        self.yl_curve = self.vertical_plot.plot(pen=pg.mkPen('b', width=1.5))