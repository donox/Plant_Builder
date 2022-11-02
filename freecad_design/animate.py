from PySide import QtCore
import FreeCADGui as Gui
import FreeCAD


def reset():
    FreeCAD.animation_reset()


def update():
    FreeCAD.animation_update()
    # update the scene
    Gui.updateGui()


def stopper():
    timer.stop()


FreeCAD.stopper = stopper

try:
    timer = QtCore.QTimer()
    # connect timer event to function "update"
    timer.timeout.connect(update)
    # start the timer to trigger "update" every 10 ms
    timer.start(50)
    if not timer.isActive():
        print("NOT RUNNING")
except Exception as e:
    print(f"FAILED: {e.args}")
