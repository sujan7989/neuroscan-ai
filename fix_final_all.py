import os, re

ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'

def rd(fp):
    with open(ROOT+fp, encoding='utf-8') as f: return f.read()

def wr(fp, s):
    with open(ROOT+fp, 'w', encoding='utf-8', newline='\n') as f: f.write(s)
    print(f"OK: {fp}")

# ── 1. server.js ─────────────────────────────────────────────────
srv = rd(r'\server\server.js')

# Replace the SPD fallback (lines 1069-1070) with SSD
srv = srv.replace(
    "// Sensory Processing Disorder (SPD)",
    "// Speech Sound Disorder (SSD)"
)
# SPD content block
old_spd = srv[srv.find("if (query.includes('sensory') ||"):srv.find("if (query.includes('sensory') ||") + 800]
if "Sensory Processing" in old_spd:
    new_ssd_block = """if (query.includes('ssd') || query.includes('speech sound') || query.includes('articulation') || query.includes('phoneme')) {
      return `### 🔊 Speech Sound Disorder (SSD)\nSSD affects ~1 in 12 children. It involves consistent errors producing specific speech sounds.\n\n**Signs:**\n• Substitutes sounds (w→r, f→th, t→k)\n• Under 50% intelligible to strangers after age 3\n• Persists beyond typical timelines\n\n**Treatment:** Articulation therapy (traditional approach) or Metaphon/Cycles phonological intervention by a certified SLP. GFTA-3 is the gold-standard assessment.\n\n⚠️ *Rule out hearing loss first — consult SLP and audiologist.*\`"""
    srv = srv[:srv.find("if (query.includes('sensory') ||")] + new_ssd_block + srv[srv.find("if (query.includes('sensory') ||") + len(old_spd):]

# Replace DLD references in doctor chat system prompt
srv = srv.replace(
    "- DLD (Developmental Language Disorder): vocabulary delays, sentence structure, SLT therapy",
    "- Stuttering (Fluency Disorder): disfluency, fluency shaping, Lidcombe programme, SEP-28k"
)
srv = srv.replace(
    "- Tourette Syndrome: motor/vocal tics, CBIT therapy, clonidine, school accommodations",
    "- SSD (Speech Sound Disorder): phoneme errors, PCC score, GFTA-3, articulation therapy"
)

# Replace topic detection
srv = srv.replace(
    "if (cMsg.includes('dld') || cMsg.includes('language disorder') || cMsg.includes('vocabulary')) userTopics.push('DLD');",
    "if (cMsg.includes('stutter') || cMsg.includes('fluency') || cMsg.includes('stammer')) userTopics.push('Stuttering');"
)
srv = srv.replace(
    "if (cMsg.includes('tourette') || cMsg.includes('tic')) userTopics.push('Tourette');",
    "if (cMsg.includes('ssd') || cMsg.includes('speech sound') || cMsg.includes('articulation') || cMsg.includes('phoneme')) userTopics.push('SSD');"
)

# Replace DLD fallback function block
old_dld_fb_start = srv.find("  // DLD\n  if (q.includes('dld')")
if old_dld_fb_start < 0:
    old_dld_fb_start = srv.find("  // DLD")
if old_dld_fb_start > 0:
    old_dld_fb_end = srv.find("  // Tourette", old_dld_fb_start)
    old_ts_fb_end  = srv.find("\n  // ", old_dld_fb_end + 100)
    if old_ts_fb_end < 0:
        old_ts_fb_end = srv.find("\n}", old_dld_fb_end + 100) + 1

    new_stutter_ssd = """  // Stuttering
  if (q.includes('stutter') || q.includes('fluency') || q.includes('stammer') || q.includes('disfluency')) {
    return `### 🎙️ Stuttering / Fluency Disorder\n\n**What is stuttering?**\nStuttering affects ~1 in 100 people and involves repetitions (b-b-ball), prolongations (ssssun), and blocks. It typically begins between ages 2-4.\n\n**Evidence-based treatment:**\n• **Lidcombe Programme** — for children under 6; parent-delivered operant therapy with strong evidence\n• **Fluency Shaping** — controlled breathing, easy onsets, smooth speech for older children/adults\n• **CAMPSIE programme** — intensive residential fluency therapy\n• **CBT** — for speaking anxiety and avoidance in adolescents and adults\n\n**Key facts:** Male 3:1 ratio; 60-80% of children recover naturally before puberty; family history increases risk 3-fold.\n\n**In India:** AIISH Mysore, NIMHANS Bangalore, Sri Ramachandra Institute Chennai.\n⚠️ *Screening only — consult a certified Speech-Language Pathologist (SLP).*\`;
  }
  // SSD
  if (q.includes('ssd') || q.includes('speech sound') || q.includes('articulation') || q.includes('phoneme') || q.includes('intelligib')) {
    return `### 🔊 Speech Sound Disorder (SSD)\n\n**What is SSD?**\nSSD (ICD F80.0) affects ~1 in 12 children and involves consistent errors producing specific speech sounds, reducing intelligibility.\n\n**Signs to watch:**\n• Consistent substitutions: w→r, f→th, t→k, d→g\n• Speech under 50% intelligible to strangers after age 3\n• Errors persist beyond typical developmental timelines\n• Phonological processes (e.g. cluster reduction, final consonant deletion)\n\n**Gold-standard assessment:** GFTA-3 (Goldman-Fristoe Test of Articulation, normative sample N=2621)\n\n**Evidence-based treatment:**\n• **Traditional Articulation Therapy** — target sounds at phoneme→syllable→word→sentence level\n• **Metaphon / Cycles Approach** — for phonological pattern errors\n• **Nuffield Dyspraxia Programme** — if motor speech planning is affected\n\n**Always rule out hearing loss first — audiological evaluation is essential.**\n⚠️ *Consult a certified SLP for formal GFTA-3 assessment.*\`;
  }
"""
    srv = srv[:old_dld_fb_start] + new_stutter_ssd + srv[old_ts_fb_end:]
    print("server.js DLD/Tourette fallbacks replaced with Stuttering/SSD")

wr(r'\server\server.js', srv)

# ── 2. assess.html ───────────────────────────────────────────────
ass = rd(r'\pages\assess.html')

# Fix CSS root variables
ass = ass.replace('--dld:#7c3aed; --ts:#dc2626;', '--stu:#0891b2; --ssd:#7c3aed;')

# Fix disorder badges
ass = ass.replace(
    '<span class="disorder-badge">&#x1F5E8; DLD</span>',
    '<span class="disorder-badge">&#x1F399; Stuttering</span>'
)
ass = ass.replace(
    '<span class="disorder-badge">&#x1F300; Tourette</span>',
    '<span class="disorder-badge">&#x1F50A; SSD</span>'
)

# Fix info bar text that uses 'dld' key
ass = ass.replace(
    "dld:      '&#x1F9EA; <strong>Screening all disorders</strong>",
    "all:      '&#x1F9EA; <strong>Screening all disorders</strong>"
)

# Fix the DISORDER_FILTER_INFO keys — dld-> stuttering, tourette -> ssd
ass = ass.replace(
    "dld:      '&#x1F5E8; <strong>Stut",
    "stuttering: '&#x1F399; <strong>Stut"
)
ass = ass.replace(
    "tourette: '&#x1F300; <strong>SSD-",
    "ssd:      '&#x1F50A; <strong>SSD-"
)
# Fix "Screening all disorders" info mentioning dld/tourette
ass = ass.replace(
    "questions cover ASD, ADHD, Dyslexia, DLD, and Tourette",
    "questions cover ASD, ADHD, Dyslexia, Stuttering, and SSD"
)
wr(r'\pages\assess.html', ass)

# ── 3. media-analysis.html ───────────────────────────────────────
med = rd(r'\pages\media-analysis.html')

# Fix header badges
med = med.replace(
    '<span class="hdr-badge">&#x1F5E8; DLD</span>',
    '<span class="hdr-badge">&#x1F399; Stuttering</span>'
)
med = med.replace(
    '<span class="hdr-badge">&#x1F300; Tourette</span>',
    '<span class="hdr-badge">&#x1F50A; SSD</span>'
)

# Fix placeholder list items
med = med.replace(
    '<div>&#x1F5E8; DLD &rarr; language structure, vocabulary, fluency</div>',
    '<div>&#x1F399; Stuttering &rarr; disfluency, repetitions, blocks, fluency</div>'
)
med = med.replace(
    '<div>&#x1F300; Tourette &rarr; tic movements, vocal patterns</div>',
    '<div>&#x1F50A; SSD &rarr; phoneme errors, articulation, intelligibility</div>'
)

# Fix RECOMMENDATIONS_DB — replace dld and tourette entries
# Find and replace dld block
dld_start = med.find("\n  dld: {")
if dld_start < 0: dld_start = med.find("\ndld: {")
ts_start  = med.find("\n  tourette: {")
if ts_start < 0: ts_start = med.find("\ntourette: {")

if dld_start > 0 and ts_start > 0:
    # Find end of tourette block (closing };)
    ts_end = med.find("\n};", ts_start) + 3
    
    new_recs = """
  stuttering: {
    label: 'Stuttering / Fluency Disorder',
    icon: '&#x1F399;',
    therapies: [
      { name: 'Fluency Shaping Therapy', detail: 'Teaches controlled breathing, easy onsets, and smooth speech. Most effective for school-age children and adults.' },
      { name: 'Lidcombe Programme', detail: 'Parent-delivered operant treatment for children under 6. Strong evidence base — significantly reduces stuttering frequency.' },
      { name: 'CBT for Stuttering', detail: 'Addresses speaking anxiety and avoidance behaviours that often develop alongside stuttering in adolescents.' },
      { name: 'CAMPSIE Intensive Programme', detail: 'Intensive residential fluency therapy delivering concentrated treatment over a short period.' }
    ],
    specialists: ['Speech-Language Pathologist (Fluency Specialist)', 'Psychologist (for anxiety)', 'Paediatric SLP'],
    hospitals: [
      { name: 'AIISH Mysore', detail: 'All India Institute of Speech and Hearing — premier fluency disorder centre in India.' },
      { name: 'NIMHANS Bangalore', detail: 'Comprehensive speech and language services including fluency disorders.' },
      { name: 'Sri Ramachandra Institute Chennai', detail: 'SLP department with fluency therapy and parent training programmes.' }
    ],
    nextSteps: [
      'Seek evaluation from a certified Speech-Language Pathologist specialising in fluency.',
      'For children under 6: start Lidcombe Programme immediately — early intervention has best outcomes.',
      'For older children: enrol in fluency shaping or stuttering modification therapy.',
      'Address speaking anxiety with a psychologist if avoidance behaviours are present.'
    ]
  },
  ssd: {
    label: 'Speech Sound Disorder (SSD)',
    icon: '&#x1F50A;',
    therapies: [
      { name: 'Traditional Articulation Therapy', detail: 'Systematic practice of target sounds at phoneme, syllable, word, and sentence levels. Most evidence-based approach for SSD.' },
      { name: 'Phonological Intervention (Metaphon)', detail: 'Targets phonological pattern errors by teaching the child the rule system underlying sound contrasts.' },
      { name: 'Cycles Approach', detail: 'Targets multiple phonological processes in rotating cycles — effective for highly unintelligible children.' },
      { name: 'Nuffield Dyspraxia Programme', detail: 'For motor speech planning difficulties — uses visual and tactile cues to establish motor programmes.' }
    ],
    specialists: ['Speech-Language Pathologist (SLP)', 'Audiologist (hearing evaluation)', 'Paediatric Neurologist (if motor speech)'],
    hospitals: [
      { name: 'AIISH Mysore', detail: 'Premier speech and hearing institute offering comprehensive GFTA-3 assessment and articulation therapy.' },
      { name: 'NIMHANS Bangalore', detail: 'Child language and speech disorders clinic with certified SLPs.' },
      { name: 'Kokilaben Dhirubhai Ambani Hospital Mumbai', detail: 'Paediatric SLP department with phonological intervention services.' }
    ],
    nextSteps: [
      'Get a formal GFTA-3 (Goldman-Fristoe Test of Articulation) assessment from a certified SLP.',
      'Rule out hearing loss — audiological evaluation is essential before starting therapy.',
      'Begin articulation or phonological therapy 2-3 sessions per week.',
      'Involve parents in home practice — 10 minutes daily of target sound practice accelerates outcomes.'
    ]
  }
};"""
    med = med[:dld_start] + new_recs + med[ts_end:]
    print("media-analysis.html RECOMMENDATIONS_DB replaced with Stuttering/SSD")
else:
    print(f"WARN: dld_start={dld_start} ts_start={ts_start}")

# Fix DISORDER_LABELS
med = med.replace(
    "dld: 'Developmental Language Disorder',",
    "stuttering: 'Stuttering / Fluency Disorder',"
)
med = med.replace(
    "tourette: 'Tourette Syndrome / Tic Disorders'",
    "ssd: 'Speech Sound Disorder'"
)

wr(r'\pages\media-analysis.html', med)

# ── 4. index.html — force correct disorder tiles ─────────────────
idx = rd(r'\index.html')
# Find disorders-grid and replace entire block
grid_s = idx.find('<div class="disorders-grid">')
if grid_s >= 0:
    # Find closing </div> of the grid
    depth = 0; pos = grid_s; grid_e = grid_s + 500
    while pos < len(idx):
        if idx[pos:pos+4] == '<div': depth += 1
        if idx[pos:pos+6] == '</div': depth -= 1
        if depth == 0 and pos > grid_s:
            grid_e = pos + 6; break
        pos += 1
    new_grid = '''<div class="disorders-grid">
      <div class="disorder-tile" style="border-color:#2d6a4f30;background:#f0faf3"><div class="disorder-tile-icon">&#x1F9E9;</div><div class="disorder-tile-name" style="color:#2d6a4f">ASD</div><div class="disorder-tile-icd">F84.0</div><div class="disorder-tile-prob" style="color:#2d6a4f">1 in 36</div></div>
      <div class="disorder-tile" style="border-color:#e76f5130;background:#fff5f2"><div class="disorder-tile-icon">&#x26A1;</div><div class="disorder-tile-name" style="color:#e76f51">ADHD</div><div class="disorder-tile-icd">F90.0</div><div class="disorder-tile-prob" style="color:#e76f51">1 in 10</div></div>
      <div class="disorder-tile" style="border-color:#457b9d30;background:#f0f6fb"><div class="disorder-tile-icon">&#x1F4D6;</div><div class="disorder-tile-name" style="color:#457b9d">Dyslexia</div><div class="disorder-tile-icd">F81.0</div><div class="disorder-tile-prob" style="color:#457b9d">1 in 5</div></div>
      <div class="disorder-tile" style="border-color:#0891b230;background:#f0fbff"><div class="disorder-tile-icon">&#x1F399;</div><div class="disorder-tile-name" style="color:#0891b2">Stuttering</div><div class="disorder-tile-icd">F98.5</div><div class="disorder-tile-prob" style="color:#0891b2">1 in 100</div></div>
      <div class="disorder-tile" style="border-color:#7c3aed30;background:#f5f0ff"><div class="disorder-tile-icon">&#x1F50A;</div><div class="disorder-tile-name" style="color:#7c3aed">SSD</div><div class="disorder-tile-icd">F80.0</div><div class="disorder-tile-prob" style="color:#7c3aed">1 in 12</div></div>
    </div>'''
    idx = idx[:grid_s] + new_grid + idx[grid_e:]
    print("index.html disorders-grid replaced")

# Fix orbit nodes
idx = idx.replace(
    '>&#x1F630;<div class="orbit-node-label">SAD</div></div>',
    '>&#x1F399;<div class="orbit-node-label">STU</div></div>'
)
idx = idx.replace(
    '>&#x1F5E3;<div class="orbit-node-label">SLD</div></div>',
    '>&#x1F50A;<div class="orbit-node-label">SSD</div></div>'
)
# Also update hero description
idx = idx.replace(
    'Social Anxiety</span>',
    'Stuttering</span>'
)
idx = idx.replace(
    'Speech Delay</span>',
    'SSD</span>'
)
idx = idx.replace(
    'Intellectual Disability</span>',
    'SSD</span>'
)
idx = idx.replace(
    'Sensory Processing</span>',
    'Stuttering</span>'
)
wr(r'\index.html', idx)

# ── 5. Final audit ────────────────────────────────────────────────
print("\n=== FINAL AUDIT ===")
OLD_D = ['dld','tourette','social_anxiety','speech_delay','intellectual_disability','spd']
NEW_D = ['asd','adhd','dyslexia','stuttering','ssd']
FILES_CHECK = [
    (r'\index.html', 'index'),
    (r'\js\ai-engine.js', 'ai-engine'),
    (r'\server\server.js', 'server'),
    (r'\pages\assess.html', 'assess'),
    (r'\pages\media-analysis.html', 'media-analysis'),
    (r'\pages\tracker.html', 'tracker'),
    (r'\pages\dataset.html', 'dataset'),
    (r'\pages\doctor-chat.html', 'doctor-chat'),
]
all_ok = True
for fp, label in FILES_CHECK:
    src = rd(fp)
    old_found = [d for d in OLD_D if re.search(rf'(?<![a-zA-Z_]){re.escape(d)}(?![a-zA-Z_])', src)]
    new_count = sum(1 for d in NEW_D if d in src)
    prob80 = '0.78' in src or '0.80' in src or '80-85' in src or 'calibrat' in src.lower()
    status = 'OK' if not old_found else 'FIX'
    if old_found: all_ok = False
    print(f"  [{status}] {label:<20} disorders={new_count}/5  old={old_found or '-'}  prob_cal={'Y' if prob80 else 'N'}")

print(f"\n{'ALL CLEAN AND READY' if all_ok else 'STILL HAS ISSUES'}")
