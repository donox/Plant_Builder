"""
Launcher for the PlantBuilder Task Panel.

Called from the AAPlantBuilder.FCMacro to show the panel in FreeCAD's
Tasks tab.
"""


def show_panel():
    """Launch the PlantBuilder task panel in FreeCAD."""
    import FreeCADGui
    from gui.task_panel import PlantBuilderPanel

    panel = PlantBuilderPanel()
    FreeCADGui.Control.showDialog(panel)
