f = r'c:\Users\sujan\Downloads\NeuroScan-Fresh\ml\inference\multi_predictor_v2.py'
with open(f, encoding='utf-8') as fh:
    src = fh.read()

# Fix stuttering: only triggered by speech-specific answers
old_stu = "        sev = float(answers.get('q_language_development',\n                    answers.get('q_speech_fluency', 0)) or 0) / 4.0"
new_stu = "        # Only use explicit speech-fluency answer; 0 if not provided (avoids ASD answers triggering stuttering)\n        sev = float(answers.get('q_speech_fluency', 0) or 0) / 4.0"

# Fix SSD: only triggered by articulation-specific answers
old_ssd = "        sev = float(answers.get('q_articulation',\n                    answers.get('q_language_development', 0)) or 0) / 4.0"
new_ssd = "        # Only use explicit articulation answer; 0 if not provided\n        sev = float(answers.get('q_articulation', 0) or 0) / 4.0"

changed = 0
if old_stu in src:
    src = src.replace(old_stu, new_stu); changed += 1; print("Fixed stuttering")
if old_ssd in src:
    src = src.replace(old_ssd, new_ssd); changed += 1; print("Fixed SSD")

if changed == 0:
    # Try without indent variations
    print("Trying regex replacement...")
    import re
    src = re.sub(
        r"sev = float\(answers\.get\('q_language_development',\s+answers\.get\('q_speech_fluency', 0\)\) or 0\) / 4\.0",
        "sev = float(answers.get('q_speech_fluency', 0) or 0) / 4.0",
        src
    )
    src = re.sub(
        r"sev = float\(answers\.get\('q_articulation',\s+answers\.get\('q_language_development', 0\)\) or 0\) / 4\.0",
        "sev = float(answers.get('q_articulation', 0) or 0) / 4.0",
        src
    )
    print("Regex done")

with open(f, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(src)
print("Written")

# Verify
with open(f, encoding='utf-8') as fh: v = fh.read()
print("q_speech_fluency (no fallback):", "q_speech_fluency', 0) or 0)" in v)
