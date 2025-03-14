import os.path
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from psychsim.ui.mainwindow import Ui_MainWindow
from psychsim.ui.worldview import WorldView
from psychsim.ui.mapview import MapView
from psychsim.world import World, loadWorld

settings = QSettings('USC ICT', 'PsychSim')


class PsychSimUI(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        self.world = None
        super(PsychSimUI, self).__init__(parent)
        self.setupUi(self)
        self.scene = WorldView(self.graphicsView)
        self.map = MapView(self.graphicsView)
        self.graphicsView.setScene(self.scene)

    @pyqtSlot()  # signal with no arguments
    def on_actionOpen_triggered(self):
        filename, types = QFileDialog.getOpenFileName(self, "PsychSim -- Open File")
        if filename:
            self.openScenario(str(filename))

    def openScenario(self, filename):
        if os.path.splitext(filename)[1] == '.psy':
            self.world = loadWorld(filename)
            for agent in self.world.agents.values():
                assert agent.world is self.world
#                    msg = QMessageBox()
#                    msg.setIcon(QMessageBox.Warning)
#                    msg.setText('Scenario file saved under different version of Python')
#                    msg.setInformativeText(filename)
#                    msg.setWindowTitle('Unable to open scenario')
#                    msg.setStandardButtons(QMessageBox.Ok)
#                    msg.exec_()
#                    return
        else:
            try:
                self.world = World(filename)
            except FileNotFoundError:
                return None
        self.scene.world = self.world
        self.scene.clear()
        settings.setValue('LastFile', os.path.abspath(filename))
        if settings.value('ViewCyclical') == 'yes':
            self.findChild(QAction, 'actionGround_Truth').setChecked(True)
            self.on_actionGround_Truth_triggered()
        else:
            self.findChild(QAction, 'actionAcyclical').setChecked(True)
            self.scene.displayWorld()

    @pyqtSlot()  # signal with no arguments
    def on_actionSave_triggered(self):
        filename = settings.value('LastFile').toString()
        self.scene.world.save(str(filename))
        self.scene.unsetDirty()

    @pyqtSlot()  # signal with no arguments
    def on_actionQuit_triggered(self):
        global app
        app.quit()

    @pyqtSlot()  # signal with no arguments
    def on_actionAgent_triggered(self):
        self.scene.colorNodes('agent')

    @pyqtSlot()  # signal with no arguments
    def on_actionLikelihood_triggered(self):
        self.scene.colorNodes('likelihood')

    @pyqtSlot()  # signal with no arguments
    def on_actionMap_triggered(self):
        self.graphicsView.setScene(self.map)

    @pyqtSlot()  # signal with no arguments
    def on_actionGround_Truth_triggered(self):
#        self.world.clearCoords()
        self.scene.world = self.world
#        self.scene.clear()
        self.scene.displayGroundTruth(maxRows=6)
        settings.setValue('ViewCyclical', 'yes')

    @pyqtSlot() # signal with no arguments
    def on_actionAcyclical_triggered(self):
        self.world.clearCoords()
        self.scene.world = self.world
        self.scene.clear()
        self.scene.displayWorld()
        settings.setValue('ViewCyclical', 'no')

    @pyqtSlot() # signal with no arguments
    def on_actionBeliefs_triggered(self):
        button = self.findChild(QAction, 'actionBeliefs')
        if self.findChild(QAction, 'actionAcyclical').isChecked():
            pass
        else:
            self.scene.displayGroundTruth(maxRows=6, recursive=button.isChecked())

    @pyqtSlot()  # signal with no arguments
    def on_actionStep_triggered(self):
        self.scene.step()

    @pyqtSlot()  # signal with no arguments
    def on_actionScreenshot_triggered(self):
        name, types = QFileDialog.getSaveFileName(self, 'Save File')
        if name:
            self.scene.saveImage(name)

    @pyqtSlot()  # signal with no arguments
    def on_actionSubgraphs_triggered(self):
        name = QFileDialog.getExistingDirectory(self, 'Destination Directory',
                                                options=QFileDialog.ShowDirsOnly)
        if name:
            self.scene.saveSubgraphs(name)

    def wheelEvent(self,event):
#        factor = 1.41**(-event.delta()/240.)
        factor = 1.41**(-event.pixelDelta().y()/240.)
        self.graphicsView.scale(factor, factor)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', default=None, nargs='?',
                        help='File containing an existing PsychSim scenario')
    parser.add_argument('-c', '--cyclical', action='store_true', help='Start with cyclical view of graph')

    global app
    app = QApplication(sys.argv)
    app.setOrganizationName('USC ICT')
    app.setOrganizationDomain('ict.usc.edu')
    app.setApplicationName('PsychSim')
    font = QFont('Open Sans')
    app.setFont(font)

    args = parser.parse_args(args=[str(el) for el in app.arguments()][1:])
    if args.cyclical:
        settings.setValue('ViewCyclic', 'yes')
        settings.sync()
    elif settings.value('ViewCylical') is None:
        settings.setValue('ViewCyclical', 'no')

    win = PsychSimUI()
    if args.scenario is None:
        filename = settings.value('LastFile')
        if filename and QFile.exists(filename):
            win.openScenario(filename)
    else:
        win.openScenario(args.scenario)
    win.showMaximized()
    app.exec_()


if __name__ == '__main__':
    main()