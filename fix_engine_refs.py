import re

f = r'c:\Users\sujan\Downloads\NeuroScan-Fresh\js\ai-engine.js'
with open(f, encoding='utf-8') as fh:
    src = fh.read()

# Fix CO_OCCURRENCE
src = src.replace(
    "asd:      { adhd:0.50, dld:0.45, tourette:0.22, dyslexia:0.25 },",
    "asd:      { adhd:0.50, stuttering:0.30, ssd:0.25, dyslexia:0.25 },"
)
src = src.replace(
    "adhd:     { asd:0.18, dyslexia:0.40, tourette:0.60, dld:0.30 },",
    "adhd:     { asd:0.18, dyslexia:0.40, stuttering:0.35, ssd:0.20 },"
)
src = src.replace(
    "dyslexia: { adhd:0.45, asd:0.12, dld:0.55 },",
    "dyslexia: { adhd:0.45, asd:0.12, ssd:0.40 },"
)

# Fix disorder arrays in questions
src = src.replace("disorder: ['asd','dld']", "disorder: ['asd','stuttering']")
src = src.replace("disorder: ['asd','social_anxiety','dld']", "disorder: ['asd','ssd']")
src = src.replace("disorder: ['asd','dld','adhd']", "disorder: ['asd','stuttering','adhd']")
src = src.replace("disorder: ['asd','adhd','dld']", "disorder: ['asd','adhd','stuttering']")
src = src.replace("disorder: ['dyslexia','dld']", "disorder: ['dyslexia','ssd']")
src = src.replace("disorder: ['adhd','asd','dld']", "disorder: ['adhd','asd','stuttering']")
src = src.replace("disorder: ['dld','asd']", "disorder: ['stuttering','asd']")
src = src.replace("disorder: ['dld']", "disorder: ['stuttering']")
src = src.replace("disorder: ['dyslexia','adhd','dld']", "disorder: ['dyslexia','adhd','ssd']")
src = src.replace("disorder: ['asd','speech_delay']", "disorder: ['asd','stuttering']")
src = src.replace("disorder: ['social_anxiety','asd']", "disorder: ['stuttering','asd']")
src = src.replace("disorder: ['intellectual_disability','asd']", "disorder: ['ssd','asd']")

# Remove dld weight entries from weights objects
src = re.sub(r",\s*dld:\s*\[[^\]]*\]", "", src)
src = re.sub(r"dld:\s*\[[^\]]*\],?\s*", "", src)

# Remove tourette weight entries
src = re.sub(r",\s*tourette:\s*\[[^\]]*\]", "", src)
src = re.sub(r"tourette:\s*\[[^\]]*\],?\s*", "", src)

with open(f, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(src)

remaining = src.count("dld:") + src.count("tourette:")
print(f"Remaining dld:/tourette: references: {remaining}")
print(f"Has stuttering: {'stuttering' in src}")
print(f"Has ssd: {'ssd' in src}")
print("Done")
