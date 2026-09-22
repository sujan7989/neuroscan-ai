"""Fix UTF-8 BOM in all HTML, JS, CSS, JSON, and Python files"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXTENSIONS = {'.html', '.js', '.css', '.json', '.py', '.md', '.txt', '.yaml', '.env'}
fixed = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    # Skip node_modules, .git, __pycache__
    dirnames[:] = [d for d in dirnames if d not in ('node_modules', '.git', '__pycache__', '.kiro')]
    for fname in filenames:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in EXTENSIONS:
            continue
        fpath = os.path.join(dirpath, fname)
        try:
            with open(fpath, 'rb') as f:
                raw = f.read()
            # Check for UTF-8 BOM (EF BB BF)
            if raw[:3] == b'\xef\xbb\xbf':
                content = raw[3:].decode('utf-8', errors='replace')
                with open(fpath, 'w', encoding='utf-8', newline='') as f:
                    f.write(content)
                rel = os.path.relpath(fpath, ROOT)
                fixed.append(rel)
        except Exception as e:
            print(f"  ERROR {fname}: {e}")

print(f"Fixed BOM in {len(fixed)} files:")
for f in fixed:
    print(f"  {f}")
if not fixed:
    print("  No BOM files found - all clean")
