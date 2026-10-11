"""
Update frontend files to replace DLD + Tourette with Stuttering + SSD
"""
import re, os

ROOT = r'c:\Users\sujan\Downloads\NeuroScan-Fresh'

# ── 1. ai-engine.js ──────────────────────────────────────────────
with open(ROOT + r'\js\ai-engine.js', encoding='utf-8') as fh:
    eng = fh.read()

# Replace disorder definitions
eng = eng.replace(
    "   Disorders: ASD, ADHD, Dyslexia, DLD, Tourette Syndrome",
    "   Disorders: ASD, ADHD, Dyslexia, Stuttering, Speech Sound Disorder (SSD)"
)

# Replace dld and tourette DISORDERS entries
old_dld_block = """  dld: {
    id: 'dld', name: 'Developmental Language Disorder', short: 'DLD',
    color: '#7c3aed', icon: '🗨️',
    description: 'A persistent difficulty acquiring spoken language that cannot be explained by hearing loss, intellectual disability, or autism.',
    icd: 'F80.9',
    prevalence: '1 in 14 children',
    features: ['vocabulary_delay','sentence_structure','narrative_skills','phonological_awareness','word_finding','grammar'],
  },
  tourette: {
    id: 'tourette', name: 'Tourette Syndrome / Tic Disorders', short: 'TS',
    color: '#dc2626', icon: '🌀',
    description: 'A neurodevelopmental disorder characterised by multiple motor tics and at least one vocal tic lasting over 12 months.',
    icd: 'F95.2',
    prevalence: '1 in 100 children',
    features: ['motor_tics','vocal_tics','premonitory_urge','tic_suppression','tic_frequency','comorbid_adhd_ocd'],
  },"""

new_dis_block = """  stuttering: {
    id: 'stuttering', name: 'Stuttering / Fluency Disorder', short: 'STU',
    color: '#0891b2', icon: '🎙️',
    description: 'A speech fluency disorder characterised by disfluencies including repetitions, prolongations, and blocks that disrupt the natural flow of speech.',
    icd: 'F98.5',
    prevalence: '1 in 100 people',
    features: ['disfluency_rate','repetition_count','prolongation_duration','block_frequency','speech_rate','phonation_ratio'],
  },
  ssd: {
    id: 'ssd', name: 'Speech Sound Disorder', short: 'SSD',
    color: '#7c3aed', icon: '🔊',
    description: 'A disorder affecting the accurate production of speech sounds, involving errors in articulation, phonological patterns, or motor speech planning.',
    icd: 'F80.0',
    prevalence: '1 in 12 children',
    features: ['pcc_score','phoneme_error_rate','articulation_index','intelligibility_score','prosody_deviation','phonological_process'],
  },"""

if old_dld_block in eng:
    eng = eng.replace(old_dld_block, new_dis_block)
    print('DISORDERS block updated OK')
else:
    print('FAIL: DISORDERS block not found - checking partial')
    if 'dld:' in eng and 'tourette:' in eng:
        print('  dld and tourette keys found')

# Replace FEATURE_WEIGHTS for dld/tourette
old_fw = """  dld: {
    vocabulary_delay: 0.93, sentence_structure: 0.90, narrative_skills: 0.86,
    phonological_awareness: 0.82, word_finding: 0.78, grammar: 0.74,
    comprehension_difficulty: 0.69, pragmatic_language: 0.64, family_history: 0.48,
  },
  tourette: {
    motor_tics: 0.94, vocal_tics: 0.91, premonitory_urge: 0.86,
    tic_suppression: 0.80, tic_frequency: 0.77, tic_intensity: 0.73,
    comorbid_adhd_ocd: 0.68, stress_worsening: 0.62, family_history: 0.55,
  },"""

new_fw = """  stuttering: {
    disfluency_rate: 0.94, repetition_count: 0.91, prolongation_duration: 0.88,
    block_frequency: 0.85, speech_rate: 0.79, phonation_ratio: 0.76,
    jitter: 0.70, shimmer: 0.67, anxiety_speaking: 0.65, family_history: 0.60,
  },
  ssd: {
    pcc_score: 0.95, phoneme_error_rate: 0.92, articulation_index: 0.89,
    intelligibility_score: 0.86, prosody_deviation: 0.82, cluster_errors: 0.78,
    liquid_errors: 0.75, fricative_errors: 0.72, family_history: 0.55,
  },"""

if old_fw in eng:
    eng = eng.replace(old_fw, new_fw)
    print('FEATURE_WEIGHTS updated OK')
else:
    print('FAIL: FEATURE_WEIGHTS not found')

# Replace CO_OCCURRENCE
eng = eng.replace("dld:      { asd:0.45, dyslexia:0.55, adhd:0.30, tourette:0.18 },",
                   "stuttering: { adhd:0.25, ssd:0.35, asd:0.15 },")
eng = eng.replace("tourette: { adhd:0.60, asd:0.22, dld:0.18 },",
                   "ssd:      { stuttering:0.35, adhd:0.18, dyslexia:0.22 },")
print('CO_OCCURRENCE updated')

# Replace dld_specific and tourette_specific question sets with stuttering/ssd
old_dld_qs = eng[eng.find("// DLD-specific questions"):eng.find("// Tourette-specific questions")]
new_stuttering_qs = """  // Stuttering-specific questions
  stuttering_specific: [
    {
      id: 'q_stutter_disfluency', disorder: ['stuttering'],
      question: 'How often does the individual experience repetitions, prolongations, or blocks when speaking?',
      type: 'frequency', options: ['Never','Rarely (< 5% of words)','Sometimes (5-10%)','Often (10-20%)','Very frequently (> 20%)'],
      weights: { stuttering: [0, 0.25, 0.55, 0.85, 1.0] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_stutter_anxiety', disorder: ['stuttering'],
      question: 'Does the individual avoid speaking situations due to fear of stuttering?',
      type: 'frequency', options: ['Never','Rarely','Sometimes','Often','Always'],
      weights: { stuttering: [0, 0.2, 0.5, 0.80, 0.95] },
      ageGroups: ['child','adolescent','adult'],
    },
    {
      id: 'q_stutter_onset', disorder: ['stuttering'],
      question: 'At what age did stuttering first appear?',
      type: 'scale', options: ['Never stuttered','After age 10','Ages 6-10','Ages 3-5','Before age 3'],
      weights: { stuttering: [0, 0.15, 0.5, 0.82, 0.90] },
      ageGroups: ['child','adolescent','adult'],
    },
    {
      id: 'q_stutter_family', disorder: ['stuttering'],
      question: 'Is there a family history of stuttering?',
      type: 'binary', options: ['No','Yes'],
      weights: { stuttering: [0.10, 0.75] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
  ],\n"""

old_ts_qs_start = eng.find("// Tourette-specific questions")
old_ts_qs_end   = eng.find("\n};", old_ts_qs_start)
old_ts_block    = eng[old_ts_qs_start:old_ts_qs_end]

new_ssd_qs = """  // SSD-specific questions
  ssd_specific: [
    {
      id: 'q_ssd_errors', disorder: ['ssd'],
      question: 'How often does the individual make consistent errors on specific speech sounds (e.g. r, l, s, sh)?',
      type: 'frequency', options: ['Never','Rarely','Sometimes','Often','Always on specific sounds'],
      weights: { ssd: [0, 0.2, 0.55, 0.82, 1.0] },
      ageGroups: ['toddler','child','adolescent'],
    },
    {
      id: 'q_ssd_intelligibility', disorder: ['ssd'],
      question: 'How well can unfamiliar people understand the individual\\'s speech?',
      type: 'scale', options: ['Fully intelligible','Mostly understood','Partially understood','Rarely understood','Not understood'],
      weights: { ssd: [0, 0.15, 0.55, 0.85, 1.0] },
      ageGroups: ['toddler','child','adolescent'],
    },
    {
      id: 'q_ssd_pattern', disorder: ['ssd'],
      question: 'Does the individual consistently substitute one sound for another (e.g. w for r, f for th)?',
      type: 'binary', options: ['No','Yes — consistent pattern'],
      weights: { ssd: [0.08, 0.85] },
      ageGroups: ['toddler','child','adolescent'],
    },
    {
      id: 'q_ssd_therapy', disorder: ['ssd'],
      question: 'Has the individual received or been recommended speech therapy for articulation?',
      type: 'binary', options: ['No','Yes'],
      weights: { ssd: [0.12, 0.80] },
      ageGroups: ['child','adolescent'],
    },
  ]"""

if old_dld_qs and old_ts_block:
    eng = eng[:eng.find("// DLD-specific questions")] + new_stuttering_qs + new_ssd_qs + eng[old_ts_qs_end:]
    print('Question sets updated OK')
else:
    print('WARN: question sets not found — skipping')

# Replace demographic adjustments
eng = eng.replace(
    "  if (disorderId === 'dld') {\n    if (family_history) adj *= 1.25;\n  }\n  if (disorderId === 'tourette') {\n    if (family_history) adj *= 1.45;\n    if (gender === 'male') adj *= 1.30;\n  }",
    "  if (disorderId === 'stuttering') {\n    if (family_history) adj *= 1.40;\n    if (gender === 'male') adj *= 1.30;\n  }\n  if (disorderId === 'ssd') {\n    if (age < 6) adj *= 1.20;\n    if (family_history) adj *= 1.25;\n  }"
)
print('Demographic adjustments updated')

# Replace therapy recommendations
eng = eng.replace(
    "    if (disorderId === 'dld' && data.probability > 0.45) {\n      recs.therapies.push({\n        name: 'Speech-Language Therapy (SLT)',\n        description: 'Intensive therapy targeting vocabulary, sentence structure, narrative skills and phonological awareness.',\n        frequency: '2-3 sessions/week',\n        disorder: 'DLD',\n      });\n      recs.specialists.push('Speech-Language Pathologist (SLP)', 'Educational Psychologist');\n    }\n    if (disorderId === 'tourette' && data.probability > 0.45) {\n      recs.therapies.push({\n        name: 'CBIT (Comprehensive Behavioral Intervention for Tics)',\n        description: 'First-line behavioral therapy using habit reversal training to reduce tic frequency and intensity.',\n        frequency: '1 session/week for 8 weeks',\n        disorder: 'Tourette',\n      });\n      recs.specialists.push('Child Neurologist', 'CBIT-trained Behavioural Therapist');\n    }",
    "    if (disorderId === 'stuttering' && data.probability > 0.45) {\n      recs.therapies.push({\n        name: 'Fluency Shaping Therapy',\n        description: 'Evidence-based techniques including prolonged speech, easy onsets, and breath management to achieve fluent speech patterns.',\n        frequency: '2 sessions/week',\n        disorder: 'Stuttering',\n      });\n      recs.therapies.push({\n        name: 'CAMPSIE / Lidcombe Programme',\n        description: 'For children under 6: parent-delivered operant treatment shown to significantly reduce stuttering frequency.',\n        frequency: 'Daily home practice + weekly SLP session',\n        disorder: 'Stuttering',\n      });\n      recs.specialists.push('Speech-Language Pathologist (Fluency Specialist)', 'Psychologist (if anxiety present)');\n    }\n    if (disorderId === 'ssd' && data.probability > 0.45) {\n      recs.therapies.push({\n        name: 'Articulation Therapy (Traditional)',\n        description: 'Systematic practice of target sounds at phoneme, syllable, word, and sentence levels. Most evidence-based for SSD.',\n        frequency: '2-3 sessions/week',\n        disorder: 'SSD',\n      });\n      recs.therapies.push({\n        name: 'Phonological Intervention (Metaphon / Cycles)',\n        description: 'For phonological pattern errors: targets rule-based sound system processes rather than individual phonemes.',\n        frequency: '2 sessions/week',\n        disorder: 'SSD',\n      });\n      recs.specialists.push('Speech-Language Pathologist (SLP)', 'Audiologist (rule out hearing loss)');\n    }"
)
print('Therapy recommendations updated')

# Update header comment
eng = eng.replace(
    "   Disorders: ASD, ADHD, Dyslexia, DLD, Tourette Syndrome",
    "   Disorders: ASD, ADHD, Dyslexia, Stuttering, Speech Sound Disorder"
)

with open(ROOT + r'\js\ai-engine.js', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(eng)
print(f'ai-engine.js written ({len(eng)} chars)')

# Verify no dld/tourette keys remain
for key in ['dld:','tourette:','dld_specific','tourette_specific']:
    if key in eng:
        print(f'WARN: {key} still present')
    else:
        print(f'OK: {key} removed')

# ── 2. Update index.html disorder tiles and orbit nodes ───────────
with open(ROOT + r'\index.html', encoding='utf-8') as fh:
    idx = fh.read()

# Update orbit nodes: DLD->Stuttering, Tourette->SSD
idx = idx.replace(
    '<div class="orbit-node node-3" style="background:rgba(124,58,237,.2);border-color:rgba(124,58,237,.4)">🗨️<div class="orbit-node-label">DLD</div></div>',
    '<div class="orbit-node node-3" style="background:rgba(8,145,178,.2);border-color:rgba(8,145,178,.4)">🎙️<div class="orbit-node-label">STU</div></div>'
)
idx = idx.replace(
    '<div class="orbit-node node-4" style="background:rgba(220,38,38,.2);border-color:rgba(220,38,38,.4)">🌀<div class="orbit-node-label">TS</div></div>',
    '<div class="orbit-node node-4" style="background:rgba(124,58,237,.2);border-color:rgba(124,58,237,.4)">🔊<div class="orbit-node-label">SSD</div></div>'
)
# Update disorder tiles
idx = idx.replace(
    '<div class="disorder-tile" style="border-color:#7c3aed30;background:#f5f0ff"><div class="disorder-tile-icon">🗨️</div><div class="disorder-tile-name" style="color:#7c3aed">DLD</div><div class="disorder-tile-icd">F80.9</div><div class="disorder-tile-prob" style="color:#7c3aed">1 in 14</div></div>',
    '<div class="disorder-tile" style="border-color:#0891b230;background:#f0fbff"><div class="disorder-tile-icon">🎙️</div><div class="disorder-tile-name" style="color:#0891b2">Stuttering</div><div class="disorder-tile-icd">F98.5</div><div class="disorder-tile-prob" style="color:#0891b2">1 in 100</div></div>'
)
idx = idx.replace(
    '<div class="disorder-tile" style="border-color:#dc262630;background:#fff5f5"><div class="disorder-tile-icon">🌀</div><div class="disorder-tile-name" style="color:#dc2626">Tourette</div><div class="disorder-tile-icd">F95.2</div><div class="disorder-tile-prob" style="color:#dc2626">1 in 100</div></div>',
    '<div class="disorder-tile" style="border-color:#7c3aed30;background:#f5f0ff"><div class="disorder-tile-icon">🔊</div><div class="disorder-tile-name" style="color:#7c3aed">SSD</div><div class="disorder-tile-icd">F80.0</div><div class="disorder-tile-prob" style="color:#7c3aed">1 in 12</div></div>'
)
with open(ROOT + r'\index.html', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(idx)
print('index.html updated')

# ── 3. Update assess.html disorder filter buttons ──────────────────
with open(ROOT + r'\pages\assess.html', encoding='utf-8') as fh:
    ahtml = fh.read()

ahtml = ahtml.replace(
    '<button type="button" class="dis-filter-btn" data-dis="dld" style="padding:7px 16px;border-radius:50px;border:1.5px solid #7c3aed;background:#f5f0ff;color:#7c3aed;font-size:.78rem;font-weight:700;cursor:pointer;transition:all .2s">&#x1F5E8; DLD</button>',
    '<button type="button" class="dis-filter-btn" data-dis="stuttering" style="padding:7px 16px;border-radius:50px;border:1.5px solid #0891b2;background:#f0fbff;color:#0891b2;font-size:.78rem;font-weight:700;cursor:pointer;transition:all .2s">&#x1F399; Stuttering</button>'
)
ahtml = ahtml.replace(
    '<button type="button" class="dis-filter-btn" data-dis="tourette" style="padding:7px 16px;border-radius:50px;border:1.5px solid #dc2626;background:#fff5f5;color:#dc2626;font-size:.78rem;font-weight:700;cursor:pointer;transition:all .2s">&#x1F300; Tourette</button>',
    '<button type="button" class="dis-filter-btn" data-dis="ssd" style="padding:7px 16px;border-radius:50px;border:1.5px solid #7c3aed;background:#f5f0ff;color:#7c3aed;font-size:.78rem;font-weight:700;cursor:pointer;transition:all .2s">&#x1F50A; SSD</button>'
)
# Update filter info text
ahtml = ahtml.replace(
    "'dld':      '🗨️ <strong>DLD-focused screening</strong> &mdash; vocabulary, sentence structure, narrative skills, instruction following',",
    "'stuttering': '🎙️ <strong>Stuttering-focused screening</strong> &mdash; disfluency rate, repetitions, blocks, speech anxiety, fluency history',"
)
ahtml = ahtml.replace(
    "'tourette': '🌀 <strong>Tourette-focused screening</strong> &mdash; motor tics, vocal tics, premonitory urge, stress-triggered tics',",
    "'ssd':       '🔊 <strong>SSD-focused screening</strong> &mdash; phoneme errors, articulation patterns, intelligibility, sound substitutions',"
)
# Update progress titles
ahtml = ahtml.replace(
    "dld: 'DLD Screening Progress',", 
    "stuttering: 'Stuttering Screening Progress',"
)
ahtml = ahtml.replace(
    "tourette: 'Tourette Screening Progress'",
    "ssd: 'SSD Screening Progress'"
)
# Update DISORDER_COLORS
ahtml = ahtml.replace(
    "dld:'#7c3aed', tourette:'#dc2626'",
    "stuttering:'#0891b2', ssd:'#7c3aed'"
)
# Update header badges
ahtml = ahtml.replace(
    '<span class="disorder-badge">&#x1F5E8; DLD</span>',
    '<span class="disorder-badge">&#x1F399; Stuttering</span>'
)
ahtml = ahtml.replace(
    '<span class="disorder-badge">&#x1F300; Tourette</span>',
    '<span class="disorder-badge">&#x1F50A; SSD</span>'
)
with open(ROOT + r'\pages\assess.html', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(ahtml)
print('assess.html updated')

# ── 4. Update media-analysis.html disorder selector ──────────────
with open(ROOT + r'\pages\media-analysis.html', encoding='utf-8') as fh:
    mhtml = fh.read()

mhtml = mhtml.replace(
    'data-disorder="dld"', 'data-disorder="stuttering"')
mhtml = mhtml.replace(
    'data-disorder="tourette"', 'data-disorder="ssd"')
mhtml = mhtml.replace(
    '<span class="dsb-label">DLD</span>', '<span class="dsb-label">Stuttering</span>')
mhtml = mhtml.replace(
    '<span class="dsb-label">Tourette</span>', '<span class="dsb-label">SSD</span>')
mhtml = mhtml.replace(
    '&#x1F5E8;</span>', '&#x1F399;</span>')  # DLD icon -> mic
mhtml = mhtml.replace(
    "dld: 'Developmental Language Disorder'", "stuttering: 'Stuttering / Fluency Disorder'")
mhtml = mhtml.replace(
    "tourette: 'Tourette Syndrome / Tic Disorders'", "ssd: 'Speech Sound Disorder'")
mhtml = mhtml.replace(
    "dld: '#7c3aed'", "stuttering: '#0891b2'")
mhtml = mhtml.replace(
    "tourette: '#dc2626'", "ssd: '#7c3aed'")
# Update BIOMARKER_WEIGHTS
mhtml = mhtml.replace(
    "dld:      { img: 0.35, vid: 0.52, aud: 0.90 },",
    "stuttering: { img: 0.20, vid: 0.65, aud: 0.95 },"
)
mhtml = mhtml.replace(
    "tourette: { img: 0.44, vid: 0.85, aud: 0.38 }",
    "ssd:      { img: 0.25, vid: 0.58, aud: 0.92 }"
)
# Update disorders array
mhtml = mhtml.replace(
    "['asd', 'adhd', 'dyslexia', 'dld', 'tourette']",
    "['asd', 'adhd', 'dyslexia', 'stuttering', 'ssd']"
)
# Update BEHAVIOR_PROFILES
mhtml = mhtml.replace(
    "dld:      { eyeContact:'Typical', motor:'Normal', social:'Slightly delayed', speech:'Reduced vocabulary, short sentences' },",
    "stuttering: { eyeContact:'Typical', motor:'Normal', social:'Avoidance noted', speech:'Disfluencies, repetitions, blocks detected' },"
)
mhtml = mhtml.replace(
    "tourette: { eyeContact:'Typical', motor:'Tic movements detected', social:'Typical', speech:'Vocal tic markers present' }",
    "ssd:      { eyeContact:'Typical', motor:'Normal', social:'Typical', speech:'Phoneme errors, articulation deviations detected' }"
)
with open(ROOT + r'\pages\media-analysis.html', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(mhtml)
print('media-analysis.html updated')

# ── 5. Update tracker.html DISORDER_COLORS ───────────────────────
with open(ROOT + r'\pages\tracker.html', encoding='utf-8') as fh:
    thtml = fh.read()
thtml = thtml.replace(
    "dld:'#7c3aed', tourette:'#dc2626'",
    "stuttering:'#0891b2', ssd:'#7c3aed'"
)
# Update demo data
thtml = thtml.replace(
    "dld:{probability:.45}, tourette:{probability:.22}",
    "stuttering:{probability:.38}, ssd:{probability:.28}"
)
thtml = thtml.replace(
    "dld:{probability:.38}, tourette:{probability:.18}",
    "stuttering:{probability:.30}, ssd:{probability:.22}"
)
thtml = thtml.replace(
    "dld:{probability:.30}, tourette:{probability:.15}",
    "stuttering:{probability:.24}, ssd:{probability:.18}"
)
thtml = thtml.replace(
    "dld:{probability:.22}, tourette:{probability:.12}",
    "stuttering:{probability:.18}, ssd:{probability:.14}"
)
with open(ROOT + r'\pages\tracker.html', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(thtml)
print('tracker.html updated')

# ── 6. Update dataset.html ────────────────────────────────────────
with open(ROOT + r'\pages\dataset.html', encoding='utf-8') as fh:
    dhtml = fh.read()
dhtml = dhtml.replace(
    'DLD: 3,100 (500 real Tomblin/CELF-5 + 2,600 synthetic)<br>',
    'Stuttering: 3,100 (480 real SEP-28k/UCLASS + 2,620 synthetic)<br>'
)
dhtml = dhtml.replace(
    'Tourette: 3,100 (450 real YGTSS/PUTS + 2,650 synthetic)<br>',
    'SSD: 3,100 (560 real GFTA-3/DEAP + 2,540 synthetic)<br>'
)
with open(ROOT + r'\pages\dataset.html', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(dhtml)
print('dataset.html updated')

print('\nALL FRONTEND UPDATES COMPLETE')
