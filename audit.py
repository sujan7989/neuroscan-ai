import os, re

ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'

OLD_DISORDERS = ['dld', 'tourette', 'social_anxiety', 'speech_delay',
                 'intellectual_disability', 'spd']
NEW_DISORDERS = ['asd', 'adhd', 'dyslexia', 'stuttering', 'ssd']

FILES = [
    ROOT + r'\index.html',
    ROOT + r'\js\ai-engine.js',
    ROOT + r'\js\shared.js',
    ROOT + r'\server\server.js',
    ROOT + r'\pages\assess.html',
    ROOT + r'\pages\media-analysis.html',
    ROOT + r'\pages\tracker.html',
    ROOT + r'\pages\dataset.html',
    ROOT + r'\pages\doctor-chat.html',
    ROOT + r'\pages\dashboard.html',
    ROOT + r'\pages\recommendations.html',
]

SKIP_PAGES = ['speech-analysis','learning','cognitive','drawing','emotion',
              'history','predict','profile','multimodal','about','auth']

print("=" * 65)
print("DISORDER AUDIT")
print("=" * 65)

issues = []
for fp in FILES:
    if not os.path.exists(fp):
        print(f"[SKIP] {os.path.basename(fp)} not found")
        continue
    fn = os.path.basename(fp)
    with open(fp, encoding='utf-8', errors='ignore') as fh:
        src = fh.read()

    # Check old disorder references
    for d in OLD_DISORDERS:
        # Find as JS key or data attribute (not just substring in comments)
        # Look for it as standalone word in JS context
        matches = re.findall(rf"(?<![a-zA-Z_]){re.escape(d)}(?![a-zA-Z_])", src)
        if matches:
            issues.append((fn, d, len(matches)))

    # Check new disorders present (for key files)
    skip = any(s in fn for s in SKIP_PAGES)
    if not skip:
        has_s = 'stuttering' in src
        has_ssd = 'ssd' in src
        has_asd = 'asd' in src
        has_adhd = 'adhd' in src
        has_dys = 'dyslexia' in src
        print(f"  {fn:<30} asd={has_asd} adhd={has_adhd} dyslexia={has_dys} stuttering={has_s} ssd={has_ssd}")

print("\n=== OLD DISORDER REFERENCES ===")
if not issues:
    print("  None found - clean!")
else:
    for fn, d, cnt in issues:
        print(f"  [{fn}] '{d}' x{cnt}")

print("\n=== DONE ===")
