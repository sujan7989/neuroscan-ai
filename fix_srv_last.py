import re

f = r'c:\Users\sujan\Downloads\NeuroScan-Fresh\server\server.js'
with open(f, encoding='utf-8') as fh:
    src = fh.read()

before = len(re.findall(r'(?<![a-zA-Z_])(?:dld|tourette|spd)(?![a-zA-Z_])', src))
print(f"Before: {before} old refs")

# Specific text replacements
fixes = [
    ("DLD** — Developmental Language Disorder: vocabulary delays, sentence structure, SLT therapy",
     "Stuttering** — Fluency disorder: disfluency, fluency shaping, Lidcombe programme"),
    ("Tourette Syndrome** — tics, CBIT therapy, clonidine, school accommodations",
     "SSD** — Speech Sound Disorder: phoneme errors, GFTA-3, articulation therapy"),
    ("DLD (Developmental Language Disorder): vocabulary delays, sentence structure, SLT therapy",
     "Stuttering (Fluency Disorder): disfluency, fluency shaping, Lidcombe programme"),
    ("Tourette Syndrome: motor/vocal tics, CBIT therapy, clonidine, school accommodations",
     "SSD (Speech Sound Disorder): phoneme errors, PCC score, GFTA-3, articulation therapy"),
    ("DLD (Developmental Language Disorder): vocabulary, sentence structure, fluency, pauses, word-finding",
     "Stuttering: disfluency rate, repetitions, prolongations, blocks, speech rate, phonation ratio"),
    ("Tourette Syndrome / Tic Disorders: involuntary repetitive movements, vocal patterns, tic signatures",
     "SSD (Speech Sound Disorder): phoneme error rate, PCC score, articulation index, intelligibility"),
    ("**CBIT** | Tourette, Tic Disorders | Weekly for 8 weeks |",
     "**Fluency Shaping** | Stuttering | 2-3x/week |"),
    ("ADHD, and DLD", "ADHD, and Stuttering"),
    ("Tourette Syndrome medications", "SSD medications"),
]

for old, new in fixes:
    if old in src:
        src = src.replace(old, new)
        print(f"  Fixed: {old[:50]}")

# Nuclear option: regex replace any remaining standalone dld/tourette/spd
src = re.sub(r'(?<![a-zA-Z_])DLD(?![a-zA-Z_])', 'Stuttering', src)
src = re.sub(r'(?<![a-zA-Z_])dld(?![a-zA-Z_])', 'stuttering', src)
src = re.sub(r'(?<![a-zA-Z_])Tourette(?![a-zA-Z_\s])', 'SSD', src)
src = re.sub(r'(?<![a-zA-Z_])tourette(?![a-zA-Z_])', 'ssd', src)
src = re.sub(r'(?<![a-zA-Z_])SPD(?![a-zA-Z_])', 'SSD', src)
src = re.sub(r'(?<![a-zA-Z_])spd(?![a-zA-Z_])', 'ssd', src)
# Preserve "Tourette Syndrome" as it might appear in free text
src = src.replace("SSD Syndrome", "Stuttering")
src = src.replace("ssd Syndrome", "stuttering")

with open(f, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(src)

after = len(re.findall(r'(?<![a-zA-Z_])(?:dld|tourette|spd)(?![a-zA-Z_])', src))
print(f"After: {after} old refs")
print("Done")
