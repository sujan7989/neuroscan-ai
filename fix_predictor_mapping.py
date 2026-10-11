f = r'c:\Users\sujan\Downloads\NeuroScan-Fresh\ml\inference\multi_predictor_v2.py'
with open(f, encoding='utf-8') as fh:
    src = fh.read()

# Find and replace the build_feature_vector function body
old_marker = "    if dis_id == 'asd':"
new_body = """    if dis_id == 'asd':
        # AQ-10 binary: 1 if symptom present (answer >= 2 on 0-4 scale)
        def aq(k):
            return 1.0 if float(answers.get(k, 0) or 0) >= 2.0 else 0.0
        vec = [
            aq('q_eye_contact'), aq('q_social_interaction'), aq('q_attention_span'),
            aq('q_routine'), aq('q_sensory_response'), aq('q_language_development'),
            aq('q_repetitive_behavior'), aq('q_emotional_regulation'),
            aq('q_special_interests'), aq('q_anxiety_social'),
            min(1.0, age / 60.0), gender, jaund, family,
        ]

    elif dis_id == 'adhd':
        inatt  = float(answers.get('q_attention_span', 0) or 0) / 4.0
        hyper  = float(answers.get('q_hyperactivity', 0) or 0) / 4.0
        impuls = float(answers.get('q_emotional_regulation', 0) or 0) / 4.0
        asrs_i = [inatt*4.0]*6
        asrs_h = [hyper*4.0, hyper*4.0, impuls*4.0, impuls*4.0, hyper*4.0, impuls*4.0]
        c_inatt = 50.0 + inatt * 40.0
        c_hyper = 50.0 + hyper * 40.0
        vec = asrs_i + asrs_h + [c_inatt, c_hyper, family, 1.0 if inatt > 0.5 else 0.0]

    elif dis_id == 'dyslexia':
        read  = float(answers.get('q_reading', 0) or 0) / 4.0
        phon  = float(answers.get('q_phonological', answers.get('q_reading', 0)) or 0) / 4.0
        spell = float(answers.get('q_spelling', answers.get('q_reading', 0)) or 0) / 4.0
        pa   = 100.0 - read * 38.0; pm  = 100.0 - phon * 35.0
        wrf  = 100.0 - read * 40.0; nwr = 100.0 - phon * 38.0
        vec = [pa, pm, wrf, nwr,
               1.0 if read > 0.4 else 0.0,   # letter_reversal
               1.0 if spell > 0.4 else 0.0,   # spelling_errors
               1.0 if read > 0.4 else 0.0,    # slow_reading_rate
               family,                          # family_history_dyslexia
               68.0 - read * 30.0,             # hw_pen_speed_mms
               max(0.1, 0.82 - read * 0.40),  # hw_stroke_consistency
               read * 0.30,                    # hw_letter_reversal_rate
               0.05 + read * 0.17]            # hw_erasure_per_word

    elif dis_id == 'stuttering':
        sev = float(answers.get('q_language_development',
                    answers.get('q_speech_fluency', 0)) or 0) / 4.0
        anx = float(answers.get('q_anxiety_social', sev * 2.5) or sev * 2.5) / 4.0
        vec = [sev * 0.32,          # disfluency_rate
               sev * 6.5,           # repetition_per_100w
               sev * 4.5,           # block_freq_per_100w
               145.0 - sev * 75.0, # speech_rate_wpm
               max(0.1, 0.84 - sev * 0.22),  # phonation_ratio
               0.5 + sev * 1.8,    # jitter_pct
               0.5 + sev * 1.4,    # shimmer_db
               int(round(sev * 4)), # stutter_severity
               int(round(anx * 4)), # anxiety_speaking
               family,              # family_history_stutter
               1.0 if age < 14 else 0.5]  # onset_childhood

    elif dis_id == 'ssd':
        sev = float(answers.get('q_articulation',
                    answers.get('q_language_development', 0)) or 0) / 4.0
        vec = [97.0 - sev * 38.0,          # pcc_score
               100.0 - sev * 32.0,          # gfta3_standard_score
               sev * 0.38,                   # stop_error_rate
               sev * 0.48,                   # fricative_error_rate
               sev * 0.45,                   # liquid_error_rate
               sev * 0.52,                   # cluster_error_rate
               max(0.05, 0.96 - sev * 0.43),# intelligibility_score
               sev * 0.45,                   # prosody_deviation
               1.0 if sev > 0.35 else 0.0,  # late_sound_acquisition
               1.0 if sev > 0.35 else 0.0,  # phonological_process
               family]                       # family_history_ssd
    else:
        vec = [0.0] * 10"""

# Find where the old if block starts
idx = src.find(old_marker)
if idx < 0:
    print("ERROR: marker not found!")
else:
    # Find end: "    return np.array(vec" line after it
    end_marker = "    return np.array(vec, dtype=np.float32).reshape(1, -1)"
    end_idx = src.find(end_marker, idx)
    if end_idx < 0:
        print("ERROR: end marker not found!")
    else:
        src = src[:idx] + new_body + "\n" + src[end_idx:]
        print(f"Replaced build_feature_vector body ({end_idx - idx} chars old -> {len(new_body)} chars new)")

with open(f, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(src)
print(f"Written {len(src)} chars")
