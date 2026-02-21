"""
FreeCAD InitGui.py — loaded automatically when src/ is on FreeCAD's module
search path at startup (e.g. after being added to AdditionalSearchPaths in
Edit > Preferences > General > Macro).

Registers the PlantBuilder workbench and commands so they are available in
every FreeCAD session without running the macro.
"""
try:
    from gui.commands import register_commands
    from gui.workbench import register_workbench
    register_commands()
    register_workbench()
except Exception as _e:
    try:
        import FreeCAD
        FreeCAD.Console.PrintWarning(
            f"PlantBuilder: failed to register workbench: {_e}\n"
        )
    except Exception:
        pass
