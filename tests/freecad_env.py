# tests/freecad_env.py
"""FreeCAD environment setup for virtual environment - CORRECTED PATHS"""
import sys
import os


def setup_freecad_environment():
    """Configure environment to work with FreeCAD Daily in virtual environment"""

    # CORRECT FreeCAD Daily paths based on discovery
    freecad_paths = [
        '/usr/lib/freecad-daily/lib',  # Main FreeCAD module location
        '/usr/lib/freecad-daily',  # Additional libraries
        '/usr/share/freecad-daily/Mod',  # FreeCAD modules/workbenches
    ]

    # Add FreeCAD paths to sys.path
    added_paths = []
    for path in freecad_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            added_paths.append(path)
            print(f"✅ Added to Python path: {path}")
        elif not os.path.exists(path):
            print(f"⚠️  Path not found: {path}")

    # Add project paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(project_root, 'src')

    for path in [project_root, src_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
            added_paths.append(path)

    print(f"📁 Project root: {project_root}")
    print(f"📁 Source path: {src_path}")
    print(f"🐍 Python executable: {sys.executable}")

    # Check if we're in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"🔧 Virtual environment: {'Yes' if in_venv else 'No'}")

    return added_paths


def verify_freecad_import():
    """Verify that FreeCAD Daily can be imported"""
    try:
        import FreeCAD
        print(f"✅ FreeCAD Daily imported successfully")
        print(f"   Version: {FreeCAD.Version()}")
        print(f"   Module location: {FreeCAD.__file__}")

        # Test basic FreeCAD functionality
        doc = FreeCAD.newDocument("TestDoc")
        print(f"   Created test document: {doc.Name}")
        FreeCAD.closeDocument("TestDoc")
        print(f"   Closed test document successfully")

        return True
    except ImportError as e:
        print(f"❌ Cannot import FreeCAD Daily: {e}")
        print(f"   Python executable: {sys.executable}")
        print(
            f"   Virtual environment: {'Yes' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No'}")
        print(f"   Current Python paths:")
        for i, path in enumerate(sys.path[:15]):  # Show first 15 paths
            print(f"     {i:2d}: {path}")
        return False
    except Exception as e:
        print(f"❌ FreeCAD import succeeded but error occurred: {e}")
        return False


if __name__ == '__main__':
    print("🔧 Setting up FreeCAD Daily environment in virtual environment...")
    print("=" * 70)
    added_paths = setup_freecad_environment()
    print(f"\n📋 Added {len(added_paths)} paths to Python environment")
    print("\n🧪 Testing FreeCAD import...")
    success = verify_freecad_import()

    if success:
        print("\n🎉 FreeCAD Daily environment setup complete!")
        print("Ready to run tests with FreeCAD integration!")
    else:
        print("\n❌ FreeCAD Daily setup failed - check paths above")