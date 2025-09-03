import importlib, sys

def reload_package(root_name: str):
    """
    Reloads a package and all its submodules that are already imported.
    Ensures deepest modules reload first.
    """
    to_reload = [name for name in sys.modules.keys()
                 if name == root_name or name.startswith(root_name + ".")]
    # deepest first so parents see fresh children
    for name in sorted(to_reload, key=lambda s: s.count("."), reverse=True):
        mod = sys.modules.get(name)
        if mod is not None:
            importlib.reload(mod)
