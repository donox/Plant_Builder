# list_available_curves.py

import sys

sys.path.append('/home/don/FreecadProjects/Macros/PyMacros/PlantBuilder/src')

import curves

print("Available curve generation functions:")
print("=" * 50)

for name in sorted(dir(curves)):
    if name.startswith('generate_') and not name.startswith('_'):
        func = getattr(curves, name)
        curve_type = name.replace('generate_', '')

        # Try to get docstring
        doc = func.__doc__ if func.__doc__ else "No documentation"
        first_line = doc.split('\n')[0].strip()

        print(f"  {curve_type:20s} - {first_line}")

print("=" * 50)
'''

## Summary:

With
these
changes:
1. ✅ **Validation happens at YAML load time** - fails fast with clear error
2. ✅ **Lists available curves** - user knows what they can use
3. ✅ **Helpful error messages** - tells user exactly what to add
4. ✅ **Graceful failure** - no mysterious errors deep in the code

The error will now look like:

Operation 0 ('spiral'): Curve type 'spiral' is not available.
Available curve types: ['arc', 'helix', 'line', 'sine']
Add 'generate_spiral()' function to curves.py
'''