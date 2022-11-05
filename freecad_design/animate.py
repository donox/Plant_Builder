from PySide import QtCore
import FreeCADGui as Gui
import FreeCAD


def reset():
    FreeCAD.animation_reset()


def update():
    FreeCAD.animation_count += 1
    FreeCAD.animation_update()
    # update the scene
    Gui.updateGui()
    if FreeCAD.animation_count >= 21:
        stopper()


def stopper():
    timer.stop()


FreeCAD.stopper = stopper

try:
    # for j in range(10):
    #     update()
    timer = QtCore.QTimer()
    # connect timer event to function "update"
    timer.timeout.connect(update)
    timer.start(100)
    if not timer.isActive():
        print("NOT RUNNING")
except Exception as e:
    print(f"FAILED: {e.args}")
