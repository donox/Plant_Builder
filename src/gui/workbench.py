"""
FreeCAD Workbench definition for PlantBuilder.
"""
from __future__ import annotations
import os

_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "icons", "plantbuilder.svg")


def register_workbench():
    """Register the PlantBuilder workbench (idempotent)."""
    import FreeCADGui as Gui

    if "PlantBuilderWorkbench" in Gui.listWorkbenches():
        return

    class PlantBuilderWorkbench(Gui.Workbench):
        MenuText = "PlantBuilder"
        ToolTip  = "Plant Builder — parametric botanical structures"
        Icon     = _ICON_PATH

        def Initialize(self):
            from gui.commands import CMD_OPEN_PANEL
            self.appendToolbar("PlantBuilder", [CMD_OPEN_PANEL])
            self.appendMenu("&PlantBuilder", [CMD_OPEN_PANEL])

        def Activated(self):
            pass

        def Deactivated(self):
            pass

        def GetClassName(self):
            return "Gui::PythonWorkbench"

    Gui.addWorkbench(PlantBuilderWorkbench())
