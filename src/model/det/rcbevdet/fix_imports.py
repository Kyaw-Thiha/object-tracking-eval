"""Script to normalize local RCBEVDet imports."""
import re
from pathlib import Path

def fix_imports(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Normalize legacy relative imports to current local package layout.
    replacements = [
        # Ops imports that previously relied on parent traversal.
        (r'from \.\.ops\.bev_pool_v2', r'from .ops.bev_pool_v2'),
        (r'from \.\.model_utils\.ops\.modules\.ms_deform_attn', r'from .ops.ms_deform_attn'),

        # Builder imports from parent directories.
        (r'from \.\.builder import', r'from .builder import'),
        (r'from \.\. import builder', r'from . import builder'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix imports in all Python files in rcbevdet directory."""
    rcbevdet_dir = Path(__file__).parent
    print(f"Fixing imports in: {rcbevdet_dir}")

    fixed_count = 0
    for py_file in rcbevdet_dir.rglob('*.py'):
        if py_file.name in ['__init__.py', 'fix_imports.py']:
            continue

        if fix_imports(py_file):
            print(f"  ✓ Fixed: {py_file.relative_to(rcbevdet_dir)}")
            fixed_count += 1
        else:
            print(f"  - Skipped: {py_file.relative_to(rcbevdet_dir)} (no changes)")

    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == '__main__':
    main()
