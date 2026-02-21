"""
FreeCAD command definitions for PlantBuilder.

Commands are registered with FreeCADGui.addCommand() so they can be
referenced by name in toolbars, menus, and keyboard shortcuts.
"""
from __future__ import annotations
import os

_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "icons", "plantbuilder.svg")

CMD_OPEN_PANEL = "PlantBuilder_OpenPanel"


class _OpenPanelCommand:
    def GetResources(self):
        return {
            "Pixmap":   _ICON_PATH,
            "MenuText": "Open PlantBuilder",
            "ToolTip":  "Open the PlantBuilder task panel",
        }

    def IsActive(self):
        return True

    def Activated(self):
        from gui.launcher import show_panel
        show_panel()


def register_commands():
    """Register all PlantBuilder commands (idempotent)."""
    import FreeCADGui as Gui
    if CMD_OPEN_PANEL not in Gui.listCommands():
        Gui.addCommand(CMD_OPEN_PANEL, _OpenPanelCommand())
