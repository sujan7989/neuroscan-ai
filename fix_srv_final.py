import re

f = r'c:\Users\sujan\Downloads\NeuroScan-Fresh\server\server.js'
with open(f, encoding='utf-8') as fh:
    srv = fh.read()

REPLACEMENTS = [
    # Knowledge base bullets
    ("• **DLD** — Developmental Language Disorder: vocabulary delays, sentence structure, SLT therapy",
     "• **Stuttering** — Fluency disorder: disfluency, fluency shaping, Lidcombe programme"),
    ("• **Tourette Syndrome** — tics, CBIT therapy, clonidine, school accommodations",
     "• **SSD** — Speech Sound Disorder: phoneme errors, GFTA-3, articulation therapy"),
    # System prompt knowledge
    ("- DLD (Developmental Language Disorder): vocabulary delays, sentence structure, SLT therapy",
     "- Stuttering (Fluency Disorder): disfluency, fluency shaping, Lidcombe programme"),
    ("- Tourette Syndrome: motor/vocal tics, CBIT therapy, clonidine, school accommodations",
     "- SSD (Speech Sound Disorder): phoneme errors, PCC score, GFTA-3, articulation therapy"),
    ("- Tourette Syndrome / Tic Disorders",
     "- SSD (Speech Sound Disorder)"),
    # Gemini prompt disorders list
    ("4. DLD (Developmental Language Disorder): vocabulary, sentence structure, fluency, pauses, word-finding",
     "4. Stuttering: disfluency rate, repetitions, prolongations, blocks, speech rate, phonation ratio"),
    ("5. Tourette Syndrome / Tic Disorders: involuntary repetitive movements, vocal patterns, tic signatures",
     "5. SSD (Speech Sound Disorder): phoneme error rate, PCC score, articulation index, intelligibility"),
    # JSON schema in prompt
    ('"dld": <integer 0-100>,',   '"stuttering": <integer 0-100>,'),
    ('"tourette": <integer 0-100>', '"ssd": <integer 0-100>'),
    # Fallback result maps
    ("dld: speechProb,", "stuttering: speechProb,"),
    ("tourette: normalProb", "ssd: normalProb"),
    ("dld: (parsed.metrics.fluency", "stuttering: (parsed.metrics.fluency"),
    # Therapy table
    ("| **CBIT** | Tourette, Tic Disorders | Weekly for 8 weeks |",
     "| **Fluency Shaping** | Stuttering | 2-3x/week |"),
    ("Many children with ASD, ADHD, and DLD",
     "Many children with ASD, ADHD, and Stuttering"),
    ("**Tourette/Tics medications** (only for severe cases):",
     "**SSD — no medication:** Focus entirely on speech therapy."),
    # NeuroScan platform description
    ("NeuroScan AI: 5-disorder ML screening (ASD, ADHD, Dyslexia, DLD, Tourette)",
     "NeuroScan AI: 5-disorder ML screening (ASD, ADHD, Dyslexia, Stuttering, SSD)"),
    ("ASD, ADHD, Dyslexia, DLD, and Tourette Syndrome",
     "ASD, ADHD, Dyslexia, Stuttering, and SSD"),
    # Fallback function checks
    ("if (msg.includes('dld') || msg.includes('language disorder') || msg.includes('vocabulary')) {",
     "if (msg.includes('stutter') || msg.includes('fluency') || msg.includes('stammer') || msg.includes('disfluency')) {"),
    ("if (msg.includes('tourette') || msg.includes('tic') || msg.includes('tics')) {",
     "if (msg.includes('ssd') || msg.includes('speech sound') || msg.includes('articulation') || msg.includes('phoneme')) {"),
    # reply text for old DLD/tourette fallbacks
    ('      reply = "**Developmental Lang', '      reply = "**Stuttering / Fluency Disorder'),
    ('      reply = "**Tourette Syndrome', '      reply = "**SSD — Speech Sound Disorder'),
    # Default reply
    ("**Disorders:** ASD symptoms, ADHD m", "**Disorders:** ASD, ADHD, Dyslexia, Stuttering, SSD"),
    # Any remaining DLD/Tourette in system prompts
    ("DLD (Developmental Language Disorder),", "Stuttering (Fluency Disorder),"),
    ("Tourette Syndrome,", "SSD (Speech Sound Disorder),"),
]

changed = 0
for old, new in REPLACEMENTS:
    if old in srv:
        srv = srv.replace(old, new)
        changed += 1
        print(f"  Fixed: {old[:55]}")

# Final catch-all for any remaining dld/tourette as JS keys
srv = re.sub(r"(?<![a-zA-Z_])dld(?![a-zA-Z_])", "stuttering", srv)
srv = re.sub(r"(?<![a-zA-Z_])tourette(?![a-zA-Z_])", "ssd", srv)

with open(f, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(srv)

remaining = len(re.findall(r'(?<![a-zA-Z_])(?:dld|tourette)(?![a-zA-Z_])', srv))
print(f"\nTotal fixes: {changed}")
print(f"Remaining old refs: {remaining}")
print("Done")
