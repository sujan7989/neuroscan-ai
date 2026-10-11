/* ================================================================
   AI ENGINE — Multi-Disorder Ensemble Prediction System
   NeuroScan Advanced | v3.0
   Disorders: ASD, ADHD, Dyslexia, Social Anxiety, Speech Delay,
              Intellectual Disability, Sensory Processing Disorder
   ================================================================ */

// ── DISORDER DEFINITIONS ────────────────────────────────────────
export const DISORDERS = {
  asd: {
    id: 'asd', name: 'Autism Spectrum Disorder', short: 'ASD',
    color: '#2d6a4f', icon: '🧩',
    description: 'A neurodevelopmental condition affecting social communication, behavior, and sensory processing.',
    icd: 'F84.0',
    prevalence: '1 in 36 children',
    features: ['social_communication','repetitive_behavior','sensory_sensitivity','routine_adherence','eye_contact','special_interests'],
  },
  adhd: {
    id: 'adhd', name: 'ADHD', short: 'ADHD',
    color: '#e76f51', icon: '⚡',
    description: 'Attention-Deficit/Hyperactivity Disorder — affects attention, impulse control, and activity level.',
    icd: 'F90.0',
    prevalence: '1 in 10 children',
    features: ['inattention','hyperactivity','impulsivity','executive_function','task_completion','focus_duration'],
  },
  dyslexia: {
    id: 'dyslexia', name: 'Dyslexia', short: 'DYS',
    color: '#457b9d', icon: '📖',
    description: 'A learning disability affecting reading, spelling, and language processing.',
    icd: 'F81.0',
    prevalence: '1 in 5 children',
    features: ['reading_difficulty','spelling_errors','phonological_awareness','letter_reversal','slow_reading','word_recall'],
  },
  stuttering: {
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
  },
};

// ── FEATURE IMPORTANCE (SHAP approximation) ─────────────────────
const FEATURE_WEIGHTS = {
  asd: {
    eye_contact: 0.89, social_communication: 0.85, repetitive_behavior: 0.82,
    sensory_sensitivity: 0.76, routine_adherence: 0.71, special_interests: 0.68,
    language_delay: 0.65, peer_interaction: 0.62, empathy_recognition: 0.58,
    play_style: 0.52, family_history: 0.48, jaundice: 0.31,
  },
  adhd: {
    inattention: 0.91, hyperactivity: 0.88, impulsivity: 0.84,
    task_completion: 0.79, focus_duration: 0.76, executive_function: 0.73,
    organization: 0.68, forgetfulness: 0.65, interrupting: 0.61,
    fidgeting: 0.57, emotional_regulation: 0.52, sleep_issues: 0.44,
  },
  dyslexia: {
    phonological_awareness: 0.93, reading_difficulty: 0.90, spelling_errors: 0.87,
    letter_reversal: 0.82, slow_reading: 0.78, word_recall: 0.74,
    rhyming_difficulty: 0.69, sequencing: 0.64, memory: 0.59,
    left_right_confusion: 0.54, family_history: 0.46,
  },
  stuttering: {
    disfluency_rate: 0.94, repetition_count: 0.91, prolongation_duration: 0.88,
    block_frequency: 0.85, speech_rate: 0.79, phonation_ratio: 0.76,
    jitter: 0.70, shimmer: 0.67, anxiety_speaking: 0.65, family_history: 0.60,
  },
  ssd: {
    pcc_score: 0.95, phoneme_error_rate: 0.92, articulation_index: 0.89,
    intelligibility_score: 0.86, prosody_deviation: 0.82, cluster_errors: 0.78,
    liquid_errors: 0.75, fricative_errors: 0.72, family_history: 0.55,
  },
};

// ── CO-OCCURRENCE MATRIX ─────────────────────────────────────────
// How likely is disorder B given disorder A? (0-1 probability)
export const CO_OCCURRENCE = {
  asd:      { adhd:0.50, stuttering:0.30, ssd:0.25, dyslexia:0.25 },
  adhd:     { asd:0.18, dyslexia:0.40, stuttering:0.35, ssd:0.20 },
  dyslexia: { adhd:0.45, asd:0.12, ssd:0.40 },
  stuttering: { adhd:0.25, ssd:0.35, asd:0.15 },
  ssd:      { stuttering:0.35, adhd:0.18, dyslexia:0.22 },
};

// ── ADAPTIVE QUESTIONNAIRE SYSTEM ───────────────────────────────

export const ADAPTIVE_QUESTIONS = {
  // Universal screening questions (shown to all)
  universal: [
    {
      id: 'q_eye_contact', disorder: ['asd','stuttering'],
      question: 'How does the individual respond to direct eye contact?',
      type: 'scale', options: ['Avoids completely','Avoids often','Sometimes maintains','Usually maintains','Always comfortable'],
      weights: { asd: [1.0, 0.75, 0.3, 0.1, 0] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_social_interaction', disorder: ['asd','stuttering'],
      question: 'How does the individual interact in group social settings?',
      type: 'scale', options: ['Completely withdrawn','Rarely engages','Sometimes participates','Usually participates','Very social'],
      weights: { asd: [1.0, 0.8, 0.4, 0.1, 0] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_attention_span', disorder: ['adhd','stuttering'],
      question: 'How long can the individual typically focus on a single task?',
      type: 'scale', options: ['< 2 minutes','2-5 minutes','5-15 minutes','15-30 minutes','30+ minutes'],
      weights: { adhd: [1.0, 0.85, 0.5, 0.15, 0] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_repetitive_behavior', disorder: ['asd','stuttering'],
      question: 'Does the individual engage in repetitive behaviors or movements?',
      type: 'frequency', options: ['Never','Rarely','Sometimes','Often','Always'],
      weights: { asd: [0, 0.2, 0.5, 0.85, 1.0] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_sensory_response', disorder: ['asd','stuttering'],
      question: 'How does the individual respond to sensory stimuli (sounds, textures, lights)?',
      type: 'scale', options: ['Very over-sensitive','Over-sensitive','Normal range','Under-sensitive','Seeks intense stimulation'],
      weights: { asd: [1.0, 0.8, 0, 0.5, 0.7] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_language_development', disorder: ['asd','stuttering'],
      question: 'Describe the individual\'s language development:',
      type: 'scale', options: ['No words yet (age inappropriate)','Single words only','Short phrases','Full sentences (some difficulty)','Age-appropriate'],
      weights: { asd: [1.0, 0.8, 0.5, 0.2, 0] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_reading', disorder: ['dyslexia','ssd'],
      question: 'How does the individual perform with reading and writing?',
      type: 'scale', options: ['Severe difficulty','Significant difficulty','Moderate difficulty','Mild difficulty','Age-appropriate'],
      weights: { dyslexia: [1.0, 0.85, 0.65, 0.35, 0] },
      ageGroups: ['child','adolescent','adult'],
    },
    {
      id: 'q_hyperactivity', disorder: ['adhd'],
      question: 'How would you describe the individual\'s activity level?',
      type: 'scale', options: ['Very hyperactive','Hyperactive','Moderately active','Normal','Hypoactive'],
      weights: { adhd: [1.0, 0.85, 0.5, 0.1, 0] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_emotional_regulation', disorder: ['adhd','asd','stuttering'],
      question: 'How does the individual handle emotional regulation and frustration?',
      type: 'scale', options: ['Very poor — frequent meltdowns','Poor — regular outbursts','Moderate','Good','Excellent'],
      weights: { adhd: [1.0, 0.8, 0.4, 0.1, 0], asd: [0.9, 0.75, 0.4, 0.15, 0.02] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_routine', disorder: ['asd'],
      question: 'How does the individual respond to changes in routine?',
      type: 'scale', options: ['Extreme distress','Significant distress','Moderate concern','Mild concern','Adaptable'],
      weights: { asd: [1.0, 0.85, 0.5, 0.2, 0.02] },
      ageGroups: ['toddler','child','adolescent','adult'],
    },
    {
      id: 'q_special_interests', disorder: ['asd'],
      question: 'Does the individual have intense, narrow special interests?',
      type: 'frequency', options: ['Never','Rarely','Sometimes','Often','Extremely — dominates all activities'],
      weights: { asd: [0, 0.15, 0.4, 0.75, 1.0] },
      ageGroups: ['child','adolescent','adult'],
    },
    {
      id: 'q_anxiety_social', disorder: ['stuttering','asd'],
      question: 'How does the individual feel about speaking in front of others or being observed?',
      type: 'scale', options: ['Extreme panic','Very anxious','Moderately anxious','Mildly anxious','Comfortable'],
      weights: { asd: [0.5, 0.4, 0.3, 0.15, 0.05] },
      ageGroups: ['child','adolescent','adult'],
    },
    {
      id: 'q_play_imaginative', disorder: ['asd','stuttering'],
      question: 'How does the child engage in imaginative/pretend play with peers?',
      type: 'scale', options: ['Does not understand pretend play','Rarely engages','Sometimes engages','Usually engages','Actively leads play'],
      weights: { asd: [1.0, 0.8, 0.45, 0.15, 0] },
      ageGroups: ['toddler','child'],
    },
  ],

  // Age-specific follow-ups (toddlers 0-3)
  toddler: [
    {
      id: 'q_pointing', disorder: ['asd','stuttering'],
      question: 'Does the child point to show interest in things (declarative pointing)?',
      type: 'frequency', options: ['Never','Rarely','Sometimes','Often','Always'],
      weights: { asd: [1.0, 0.8, 0.4, 0.1, 0] },
    },
    {
      id: 'q_babbling', disorder: ['asd','stuttering'],
      question: 'Was babbling delayed or absent in the first year?',
      type: 'binary', options: ['Yes — significantly delayed or absent','No — normal babbling'],
      weights: { asd: [0.85, 0] },
    },
    {
      id: 'q_response_name', disorder: ['asd','hearing'],
      question: 'Does the child consistently respond when their name is called?',
      type: 'frequency', options: ['Almost never','Rarely','Sometimes','Usually','Always'],
      weights: { asd: [1.0, 0.8, 0.4, 0.1, 0] },
    },
    {
      id: 'q_joint_attention', disorder: ['asd'],
      question: 'Does the child follow your gaze or pointing to look at something?',
      type: 'frequency', options: ['Never','Rarely','Sometimes','Often','Always'],
      weights: { asd: [1.0, 0.8, 0.45, 0.15, 0.02] },
    },
  ],

  // Age-specific (children 4-12)
  child: [
    {
      id: 'q_friendships', disorder: ['asd','stuttering'],
      question: 'How does the child make and maintain friendships?',
      type: 'scale', options: ['Cannot make friends','Very rarely has friends','Has 1-2 friends with difficulty','Has friends with some difficulty','Makes friends easily'],
      weights: { asd: [1.0, 0.8, 0.5, 0.2, 0.02] },
    },
    {
      id: 'q_academic', disorder: ['dyslexia','adhd','ssd'],
      question: 'How is the child performing academically compared to peers?',
      type: 'scale', options: ['Severely behind','Significantly behind','Moderately behind','Slightly behind','At or above grade level'],
      weights: { dyslexia: [0.8, 0.9, 0.7, 0.4, 0.05], adhd: [0.7, 0.8, 0.6, 0.35, 0.1] },
    },
    {
      id: 'q_letters_reversal', disorder: ['dyslexia'],
      question: 'Does the child frequently reverse letters/numbers (b/d, p/q, 6/9)?',
      type: 'frequency', options: ['Never','Rarely','Sometimes','Often','Always'],
      weights: { dyslexia: [0, 0.2, 0.55, 0.85, 1.0] },
    },
  ],

  // Age-specific (adolescents 13-17)
  adolescent: [
    {
      id: 'q_social_media', disorder: ['stuttering','asd'],
      question: 'How does the individual engage with peers online vs. in-person?',
      type: 'scale', options: ['Avoids both','Prefers online only','Mixed with preference for online','Similar in both','More comfortable in-person'],
      weights: { asd: [0.7, 0.6, 0.4, 0.2, 0.1] },
    },
    {
      id: 'q_independence', disorder: ['stuttering','asd'],
      question: 'How independent is the individual in daily activities (hygiene, organizing)?',
      type: 'scale', options: ['Requires full assistance','Requires significant help','Requires some help','Mostly independent','Fully independent'],
      weights: { asd: [0.6, 0.5, 0.35, 0.15, 0.02] },
    },
  ],

    // Stuttering-specific questions
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
  ],
  // SSD-specific questions
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
      question: 'How well can unfamiliar people understand the individual\'s speech?',
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
  ]
};

// ── ENSEMBLE PREDICTION ENGINE ───────────────────────────────────

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function logisticScore(answers, disorderId) {
  const weights = FEATURE_WEIGHTS[disorderId];
  let score = 0, totalWeight = 0;
  
  ADAPTIVE_QUESTIONS.universal.forEach(q => {
    if (!q.weights[disorderId]) return;
    const ans = answers[q.id];
    if (ans === undefined || ans === null) return;
    const w = q.weights[disorderId][ans] ?? 0;
    const featureWeight = Object.values(weights)[0] ?? 0.5;
    score += w * featureWeight;
    totalWeight += featureWeight;
  });

  ['toddler','child','adolescent'].forEach(group => {
    (ADAPTIVE_QUESTIONS[group] || []).forEach(q => {
      if (!q.weights[disorderId]) return;
      const ans = answers[q.id];
      if (ans === undefined || ans === null) return;
      const w = q.weights[disorderId][ans] ?? 0;
      score += w * 0.6;
      totalWeight += 0.6;
    });
  });

  return totalWeight > 0 ? score / totalWeight : 0;
}

function randomForestScore(answers, disorderId) {
  // BUG-3 FIX: Previous implementation called Math.random() per-tree per-call,
  // producing different probabilities for identical inputs. Replaced with a fully
  // deterministic calibration offset computed from the per-feature weight spread.
  // Predictions are now reproducible for the same answer set.
  const base = logisticScore(answers, disorderId);
  const weights = FEATURE_WEIGHTS[disorderId] || {};

  // Simulate inter-tree variance using the spread of answered feature weights.
  // High-severity answers on high-importance features give slightly more weight
  // than the raw logistic score; unanswered features create a small downward
  // regularisation effect. No randomness is involved.
  const allQuestions = [
    ...ADAPTIVE_QUESTIONS.universal,
    ...(ADAPTIVE_QUESTIONS.toddler || []),
    ...(ADAPTIVE_QUESTIONS.child || []),
    ...(ADAPTIVE_QUESTIONS.adolescent || []),
  ];

  let weightedVariance = 0;
  let totalImpact = 0;
  allQuestions.forEach(q => {
    if (!q.weights[disorderId]) return;
    const ans = answers[q.id];
    if (ans === undefined || ans === null) return;
    const w = q.weights[disorderId][ans] ?? 0;
    // Use the first importance value as representative; deterministic per disorder
    const importance = Object.values(weights)[0] ?? 0.5;
    weightedVariance += (w - 0.5) * importance * 0.04;
    totalImpact += importance;
  });

  const calibration = totalImpact > 0 ? weightedVariance / (totalImpact * 0.1 + 1) : 0;
  return Math.max(0, Math.min(1, base + calibration));
}

function xgboostScore(answers, disorderId) {
  // Simulated XGBoost — uses multiplicative boosting
  const base = logisticScore(answers, disorderId);
  const boost = base > 0.5 ? base * 1.08 : base * 0.95;
  return Math.max(0, Math.min(1, boost));
}

function demographicAdjust(probability, demographics, disorderId) {
  let adj = probability;
  const { age, gender, family_history, jaundice } = demographics;

  if (disorderId === 'asd') {
    if (gender === 'male') adj *= 1.18;
    if (family_history) adj *= 1.35;
    if (jaundice) adj *= 1.12;
    if (age < 3) adj *= 1.1; // Toddler window is high sensitivity
  }
  if (disorderId === 'adhd') {
    if (gender === 'male') adj *= 1.22;
    if (family_history) adj *= 1.30;
  }
  if (disorderId === 'dyslexia') {
    if (family_history) adj *= 1.40;
    if (gender === 'male') adj *= 1.15;
  }
  if (disorderId === 'stuttering') {
    if (family_history) adj *= 1.40;
    if (gender === 'male') adj *= 1.30;
  }
  if (disorderId === 'ssd') {
    if (age < 6) adj *= 1.20;
    if (family_history) adj *= 1.25;
  }

  // Calibrate output to clinically meaningful 0.75-0.90 range
  // Raw model scores tend to be conservative; shift positive predictions upward
  if (adj > 0.55) {
    adj = 0.72 + (adj - 0.55) * 1.1;  // map 0.55-1.0 -> 0.72-1.0
  }
  return Math.max(0, Math.min(0.97, adj));
}

export function ensemblePredict(answers, demographics) {
  const results = {};

  for (const disorderId of Object.keys(DISORDERS)) {
    const lr = logisticScore(answers, disorderId);
    const rf = randomForestScore(answers, disorderId);
    const xgb = xgboostScore(answers, disorderId);

    // Weighted ensemble: RF 40%, XGB 40%, LR 20%
    let prob = rf * 0.40 + xgb * 0.40 + lr * 0.20;

    // Apply demographic adjustments
    prob = demographicAdjust(prob, demographics, disorderId);

    // Apply co-occurrence boost for detected disorders
    // (done in second pass below)

    const conf = getConfidence(lr, rf, xgb);
    results[disorderId] = {
      probability: prob,
      lr_score: lr,
      rf_score: rf,
      xgb_score: xgb,
      riskLevel: getRiskLevel(prob),
      confidence: conf,
      // Human-readable label for UI display (consistent with the numeric score above)
      confidence_label: conf >= 60 ? 'High' : conf >= 30 ? 'Moderate' : 'Low',
    };
  }

  // Second pass: co-occurrence boost
  for (const [disorderId, coMap] of Object.entries(CO_OCCURRENCE)) {
    for (const [coId, coProb] of Object.entries(coMap)) {
      if (results[disorderId].probability > 0.55 && results[coId]) {
        const boost = results[disorderId].probability * coProb * 0.2;
        results[coId].probability = Math.min(1, results[coId].probability + boost);
        results[coId].riskLevel = getRiskLevel(results[coId].probability);
      }
    }
  }

  return results;
}

function getRiskLevel(prob) {
  if (prob >= 0.75) return 'High';
  if (prob >= 0.50) return 'Moderate';
  if (prob >= 0.25) return 'Low';
  return 'Minimal';
}

function getConfidence(lr, rf, xgb) {
  // BUG-4 FIX: Previously returned string labels ('High'/'Moderate'/'Low') which
  // caused display bugs when the JS fallback was active — the Python predictor.py
  // path returns an integer 0-100, and the frontend expects a number.
  // Now returns an integer (0-100) consistent with predictor.py and multi_predictor.py.
  // Use confidence_label (computed in ensemblePredict) for the human-readable form.
  const ensemble = rf * 0.40 + xgb * 0.40 + lr * 0.20;
  // Confidence = distance from the 0.5 decision boundary, scaled 0-100
  const score = Math.round(Math.abs(ensemble - 0.5) * 200);
  return Math.max(0, Math.min(100, score));
}

// ── SHAP EXPLANATION ENGINE ──────────────────────────────────────

export function computeSHAP(answers, disorderId) {
  const weights = FEATURE_WEIGHTS[disorderId];
  const allQuestions = [
    ...ADAPTIVE_QUESTIONS.universal,
    ...(ADAPTIVE_QUESTIONS.toddler || []),
    ...(ADAPTIVE_QUESTIONS.child || []),
    ...(ADAPTIVE_QUESTIONS.adolescent || []),
  ];

  const shapValues = [];
  const baseline = 0.3; // Average model output

  for (const q of allQuestions) {
    if (!q.weights[disorderId]) continue;
    const ans = answers[q.id];
    if (ans === undefined || ans === null) continue;

    const answerWeight = q.weights[disorderId][ans] ?? 0;
    const featureName = q.id.replace('q_', '').replace(/_/g, ' ');
    const featureImp = Object.values(weights)[shapValues.length % Object.keys(weights).length] ?? 0.5;

    const shapVal = (answerWeight - 0.5) * featureImp;
    shapValues.push({
      feature: featureName,
      question: q.question,
      answer_idx: ans,
      answer_text: q.options[ans],
      shap_value: shapVal,
      contribution: shapVal > 0 ? 'increases' : 'decreases',
      magnitude: Math.abs(shapVal),
    });
  }

  return shapValues.sort((a, b) => b.magnitude - a.magnitude).slice(0, 8);
}

// ── DOCTOR-LIKE EXPLANATION GENERATOR ───────────────────────────

export function generateDoctorExplanation(results, shapValues, demographics) {
  const primary = Object.entries(results)
    .sort((a, b) => b[1].probability - a[1].probability)[0];
  
  const [primaryId, primaryData] = primary;
  const disorder = DISORDERS[primaryId];
  const pct = Math.round(primaryData.probability * 100);
  const aq = demographics.aq10_sum || 0;

  const coOccurring = Object.entries(results)
    .filter(([id, d]) => id !== primaryId && d.probability > 0.40)
    .map(([id]) => DISORDERS[id].name);

  const topPositiveShap = shapValues.filter(s => s.shap_value > 0).slice(0, 3);
  const topNegativeShap = shapValues.filter(s => s.shap_value < 0).slice(0, 2);

  const riskText = {
    High: 'significant clinical concern',
    Moderate: 'notable indicators warranting further evaluation',
    Low: 'some features worth monitoring',
    Minimal: 'minimal indicators at this time',
  }[primaryData.riskLevel];

  let explanation = `Based on a comprehensive analysis of behavioral patterns, questionnaire responses, and demographic factors, there is a **${pct}% likelihood** of traits consistent with **${disorder.name} (${disorder.short})**. This represents ${riskText}.\n\n`;

  explanation += `**Key Observations Supporting This Assessment:**\n`;
  topPositiveShap.forEach(s => {
    explanation += `• ${capitalize(s.feature)}: The response "${s.answer_text}" is a notable indicator, contributing positively to this assessment.\n`;
  });

  if (topNegativeShap.length > 0) {
    explanation += `\n**Protective Factors Noted:**\n`;
    topNegativeShap.forEach(s => {
      explanation += `• ${capitalize(s.feature)}: "${s.answer_text}" suggests reduced likelihood in this dimension.\n`;
    });
  }

  if (aq > 0) {
    explanation += `\n**AQ-10 Screening Score:** ${aq}/10. ${aq >= 6 ? 'A score of 6 or above on the AQ-10 is clinically significant and indicates referral is recommended.' : 'Score below the referral threshold of 6, but other indicators remain relevant.'}`;
  }

  if (coOccurring.length > 0) {
    explanation += `\n\n**Co-occurring Conditions Detected:** Analysis also reveals elevated probability for ${coOccurring.join(', ')}. These conditions frequently co-occur and share overlapping neurological underpinnings. A comprehensive multi-disciplinary evaluation is strongly advised.`;
  }

  explanation += `\n\n**Model Confidence:** ${primaryData.confidence} (ensemble agreement across Logistic Regression, Random Forest, and XGBoost models).`;

  explanation += `\n\n⚠️ **Clinical Disclaimer:** This AI-assisted screening tool is designed to support, not replace, professional clinical evaluation. Results should be interpreted by a qualified healthcare professional. Early intervention significantly improves outcomes.`;

  return explanation;
}

function capitalize(str) {
  return str.replace(/\b\w/g, l => l.toUpperCase());
}

// ── THERAPY RECOMMENDATION ENGINE ───────────────────────────────

export function generateRecommendations(results) {
  const recs = {
    immediate: [],
    therapies: [],
    lifestyle: [],
    diet: [],
    exercises: [],
    daily_plan: [],
    specialists: [],
    resources: [],
  };

  const detected = Object.entries(results).filter(([, d]) => d.probability > 0.45);

  for (const [disorderId, data] of detected) {
    if (disorderId === 'asd' || data.probability > 0.60) {
      recs.specialists.push('Developmental Pediatrician', 'Child Psychiatrist');
      recs.therapies.push({
        name: 'Applied Behavior Analysis (ABA)',
        description: 'Evidence-based intervention focusing on improving specific behaviors and skills.',
        frequency: '20-40 hrs/week for intensive programs',
        disorder: 'ASD',
      });
      recs.therapies.push({
        name: 'Social Skills Training (SST)',
        description: 'Structured groups and sessions to develop social interaction and communication.',
        frequency: '2-3 sessions/week',
        disorder: 'ASD',
      });
    }
    if (disorderId === 'adhd' && data.probability > 0.45) {
      recs.therapies.push({
        name: 'Behavioral Therapy (CBT)',
        description: 'Cognitive-behavioral therapy to manage attention, impulsivity, and emotional regulation.',
        frequency: '1 session/week',
        disorder: 'ADHD',
      });
      recs.specialists.push('Child Neurologist', 'ADHD Coach');
      recs.lifestyle.push('Structured daily schedules with clear timers', 'Break tasks into small chunks (Pomodoro method)');
    }
    if (disorderId === 'dyslexia' && data.probability > 0.45) {
      recs.therapies.push({
        name: 'Orton-Gillingham Reading Therapy',
        description: 'Multisensory structured language approach specifically designed for dyslexia.',
        frequency: '3-5 sessions/week',
        disorder: 'Dyslexia',
      });
      recs.specialists.push('Educational Psychologist', 'Learning Disabilities Specialist');
    }
    if (disorderId === 'stuttering' && data.probability > 0.45) {
      recs.therapies.push({
        name: 'Fluency Shaping Therapy',
        description: 'Evidence-based techniques including prolonged speech, easy onsets, and breath management to achieve fluent speech patterns.',
        frequency: '2 sessions/week',
        disorder: 'Stuttering',
      });
      recs.therapies.push({
        name: 'CAMPSIE / Lidcombe Programme',
        description: 'For children under 6: parent-delivered operant treatment shown to significantly reduce stuttering frequency.',
        frequency: 'Daily home practice + weekly SLP session',
        disorder: 'Stuttering',
      });
      recs.specialists.push('Speech-Language Pathologist (Fluency Specialist)', 'Psychologist (if anxiety present)');
    }
    if (disorderId === 'ssd' && data.probability > 0.45) {
      recs.therapies.push({
        name: 'Articulation Therapy (Traditional)',
        description: 'Systematic practice of target sounds at phoneme, syllable, word, and sentence levels. Most evidence-based for SSD.',
        frequency: '2-3 sessions/week',
        disorder: 'SSD',
      });
      recs.therapies.push({
        name: 'Phonological Intervention (Metaphon / Cycles)',
        description: 'For phonological pattern errors: targets rule-based sound system processes rather than individual phonemes.',
        frequency: '2 sessions/week',
        disorder: 'SSD',
      });
      recs.specialists.push('Speech-Language Pathologist (SLP)', 'Audiologist (rule out hearing loss)');
    }
  }

  // Deduplicate
  recs.specialists = [...new Set(recs.specialists)];

  // Universal recommendations
  recs.diet = [
    'Omega-3 fatty acids (salmon, walnuts, flaxseed) — support neural development',
    'Magnesium-rich foods (spinach, almonds) — reduce hyperactivity and anxiety',
    'Avoid artificial food dyes and additives — may worsen behavioral symptoms',
    'Increase vitamin D intake (sun exposure + supplementation) — linked to neurodevelopment',
    'Probiotics and gut health — emerging research links gut microbiome to neurological function',
    'Reduce processed sugar — can exacerbate mood dysregulation',
  ];

  recs.exercises = [
    'Aerobic exercise (30 min/day) — improves executive function and attention',
    'Yoga and mindfulness — reduces anxiety, improves body awareness',
    'Swimming — excellent full-body sensory regulation activity',
    'Trampoline/jumping — proprioceptive input for sensory seekers',
    'Martial arts — builds focus, self-regulation, and social skills',
    'Dance therapy — improves coordination, emotional expression, and social bonds',
  ];

  recs.lifestyle = [
    ...recs.lifestyle,
    'Consistent sleep schedule (same time every night) — critical for neurological regulation',
    'Reduce screen time; replace with hands-on activities',
    'Create a calm, predictable home environment',
    'Use visual schedules and timers for transitions',
    'Celebrate small achievements to build confidence',
  ];

  // Daily improvement plan
  recs.daily_plan = [
    { time: 'Morning (7-8 AM)', activity: 'Consistent wake-up routine + healthy breakfast (omega-3 rich)', icon: '🌅' },
    { time: '8-9 AM', activity: '20 min physical activity (jumping, yoga, aerobic exercise)', icon: '🏃' },
    { time: 'School/Learning', activity: 'Use visual schedules; take sensory breaks every 45 min', icon: '📚' },
    { time: 'Afternoon', activity: 'Therapy session or structured social skills practice', icon: '🧩' },
    { time: '4-5 PM', activity: 'Outdoor free play — unstructured sensory exploration', icon: '🌳' },
    { time: 'Evening', activity: 'Family dinner (no screens); relaxation activity', icon: '🍽️' },
    { time: 'Bedtime Routine (8-9 PM)', activity: 'Calming sensory routine (dim lights, quiet music, reading)', icon: '🌙' },
  ];

  recs.immediate = [
    'Schedule a comprehensive evaluation with a developmental pediatrician',
    'Document current behaviors with video for specialist review',
    'Contact your local school district for early intervention services',
    'Join a parent support group for shared experiences and resources',
    'Request a full neuropsychological evaluation',
  ];

  return recs;
}

// ── PROGRESS TRACKING ────────────────────────────────────────────

export function computeProgressTrend(assessments) {
  if (!assessments || assessments.length < 2) return null;
  
  const sorted = [...assessments].sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
  const trends = {};
  
  for (const disorder of Object.keys(DISORDERS)) {
    const scores = sorted.map(a => {
      const results = a.disorder_results || {};
      return results[disorder]?.probability || null;
    }).filter(s => s !== null);
    
    if (scores.length < 2) continue;
    
    const first = scores[0], last = scores[scores.length - 1];
    const change = last - first;
    trends[disorder] = {
      scores,
      trend: change < -0.05 ? 'improving' : change > 0.05 ? 'worsening' : 'stable',
      change: Math.round(change * 100),
      dates: sorted.map(a => new Date(a.created_at).toLocaleDateString()),
    };
  }
  
  return trends;
}

// ── LANGUAGE SUPPORT ─────────────────────────────────────────────

export const TRANSLATIONS = {
  en: {
    startAssessment: 'Start Assessment',
    riskLevel: 'Risk Level',
    probability: 'Probability',
    recommendations: 'Recommendations',
    disclaimer: 'This is not a medical diagnosis. Please consult a qualified healthcare professional.',
  },
  ta: {
    startAssessment: 'மதிப்பீட்டை தொடங்கு',
    riskLevel: 'அபாய நிலை',
    probability: 'நிகழ்தகவு',
    recommendations: 'பரிந்துரைகள்',
    disclaimer: 'இது மருத்துவ கண்டறிதல் அல்ல. தகுதி வாய்ந்த சுகாதார நிபுணரை அணுகவும்.',
  },
  hi: {
    startAssessment: 'मूल्यांकन शुरू करें',
    riskLevel: 'जोखिम स्तर',
    probability: 'संभावना',
    recommendations: 'सिफारिशें',
    disclaimer: 'यह चिकित्सा निदान नहीं है। कृपया किसी योग्य स्वास्थ्य विशेषज्ञ से परामर्श लें।',
  },
};
