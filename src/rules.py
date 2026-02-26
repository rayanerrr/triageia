"""
ESI-inspired rule engine for triage safety overrides.
34 rules: 14 Level 1, 12 Level 2, 8 Level 3.
Returns (level, reasons) or (None, []) if no rule fires.
"""


def rule_based_triage(v):
    """
    Apply safety rules in priority order (L1 > L2 > L3).
    v: dict with patient vitals and symptoms.
    Returns (level, reasons) or (None, []) if no override.
    """
    spo2 = v.get("spo2")
    rr = v.get("resp_rate")
    sbp = v.get("sbp")
    hr = v.get("hr")
    temp = v.get("temp")
    age = v.get("age", 50)
    confusion = v.get("confusion", 0)
    sob_sev = v.get("sob_severity", 0)
    cp_sev = v.get("chest_pain_severity", 0)
    cardiac = v.get("comorbidity_cardiac", 0)
    respiratory = v.get("comorbidity_respiratory", 0)
    diabetes = v.get("comorbidity_diabetes", 0)
    num_comorbidities = v.get("num_comorbidities", 0)
    pain = v.get("pain", 0)

    # Backward compat: binary chest_pain/sob -> severity
    if cp_sev == 0 and v.get("chest_pain", 0) == 1:
        cp_sev = 2
    if sob_sev == 0 and v.get("sob", 0) == 1:
        sob_sev = 2

    reasons_l1 = []
    reasons_l2 = []
    reasons_l3 = []

    is_infant = age <= 1
    is_child = age <= 12

    # ================================================================
    # LEVEL 1 — Immediate / Life-threatening (14 rules)
    # ================================================================

    # R1: Confusion / altered mental status
    if confusion == 1:
        reasons_l1.append("Confusion / alteration de l'etat mental")

    # R2: SpO2 < 90 — severe hypoxia
    if spo2 is not None and spo2 < 90:
        reasons_l1.append(f"SpO2={spo2}% < 90 (hypoxie severe)")

    # R3: Systolic BP < 90 — shock
    if sbp is not None and sbp < 90 and not is_child:
        reasons_l1.append(f"PAS={sbp} < 90 mmHg (hypotension / choc)")

    # R4: Severe bradycardia < 40
    if hr is not None and hr < 40:
        reasons_l1.append(f"FC={hr} < 40 (bradycardie severe)")

    # R5: Respiratory depression (RR < 6)
    if rr is not None and rr < 6:
        reasons_l1.append(f"FR={rr} < 6 (depression respiratoire)")

    # R6: Hypothermia < 33.5
    if temp is not None and temp < 33.5:
        reasons_l1.append(f"Temp={temp} < 33.5 (hypothermie severe)")

    # R7: Malignant hyperthermia >= 41
    if temp is not None and temp >= 41.0:
        reasons_l1.append(f"Temp={temp} >= 41 (hyperthermie maligne)")

    # R8: Severe dyspnea (severity 3) with objective signs
    if sob_sev >= 3 and spo2 is not None and spo2 < 94:
        reasons_l1.append("Dyspnee severe (sev=3) + SpO2 < 94")

    # R9: Severe chest pain (severity 3) with hemodynamic instability
    if cp_sev >= 3 and (
        (hr is not None and (hr > 130 or hr < 45))
        or (sbp is not None and sbp < 100)
    ):
        reasons_l1.append("Douleur thoracique severe + instabilite hemodynamique")

    # R10: Septic triad — fever + tachycardia + tachypnea + hypotension
    if (temp is not None and temp >= 38.5
            and hr is not None and hr > 110
            and rr is not None and rr > 24
            and sbp is not None and sbp < 100):
        reasons_l1.append("Triade septique (fievre + tachycardie + tachypnee + hypotension)")

    # R11: Infant — SpO2 < 92
    if is_infant and spo2 is not None and spo2 < 92:
        reasons_l1.append(f"Nourrisson SpO2={spo2} < 92")

    # R12: Infant — HR < 80 or > 200
    if is_infant and hr is not None and (hr < 80 or hr > 200):
        reasons_l1.append(f"Nourrisson FC={hr} hors limites (80-200)")

    # R13: Infant — RR < 20 or > 60
    if is_infant and rr is not None and (rr < 20 or rr > 60):
        reasons_l1.append(f"Nourrisson FR={rr} hors limites (20-60)")

    # R14: Chest pain + cardiac history + hemodynamic instability
    if cp_sev >= 2 and cardiac == 1 and (
        (hr is not None and (hr > 120 or hr < 50))
        or (sbp is not None and sbp < 100)
    ):
        reasons_l1.append("DT + ATCD cardiaque + instabilite hemodynamique")

    if reasons_l1:
        return 1, reasons_l1

    # ================================================================
    # LEVEL 2 — Urgent (12 rules)
    # ================================================================

    # R15: Tachycardia > 140
    if hr is not None and hr > 140:
        reasons_l2.append(f"FC={hr} > 140 (tachycardie severe)")

    # R16: Respiratory rate > 30
    if rr is not None and rr > 30:
        reasons_l2.append(f"FR={rr} > 30 (detresse respiratoire)")

    # R17: Temperature >= 40
    if temp is not None and temp >= 40.0:
        reasons_l2.append(f"Temp={temp} >= 40 (hyperthermie)")

    # R18: SpO2 90-91
    if spo2 is not None and 90 <= spo2 <= 91:
        reasons_l2.append(f"SpO2={spo2}% (hypoxie moderee)")

    # R19: Tachycardia + hypotension
    if hr is not None and hr > 110 and sbp is not None and sbp < 100:
        reasons_l2.append(f"Tachycardie (FC={hr}) + hypotension (PAS={sbp})")

    # R20: Fever + tachycardia (sepsis suspicion)
    if temp is not None and temp >= 38.5 and hr is not None and hr > 110:
        reasons_l2.append("Fievre + tachycardie (suspicion sepsis)")

    # R21: Moderate chest pain + cardiac history
    if cp_sev >= 2 and cardiac == 1:
        reasons_l2.append("DT moderee/severe + ATCD cardiaque")

    # R22: Moderate dyspnea (sev 2+) with respiratory compromise
    if sob_sev >= 2 and spo2 is not None and spo2 < 95:
        reasons_l2.append(f"Dyspnee moderee + SpO2={spo2}")

    # R23: Bradycardia < 45
    if hr is not None and 40 <= hr < 45:
        reasons_l2.append(f"FC={hr} < 45 (bradycardie)")

    # R24: Bradypnea < 8
    if rr is not None and 6 <= rr < 8:
        reasons_l2.append(f"FR={rr} < 8 (bradypnee)")

    # R25: Moderate hypothermia 33.5-35
    if temp is not None and 33.5 <= temp <= 35.0:
        reasons_l2.append(f"Temp={temp} (hypothermie moderee)")

    # R26: Pediatric tachycardia (child with HR > age-adjusted limit)
    if is_child and not is_infant and hr is not None:
        limit = 160 if age <= 5 else 140
        if hr > limit:
            reasons_l2.append(f"Tachycardie pediatrique FC={hr} > {limit}")

    if reasons_l2:
        return 2, reasons_l2

    # ================================================================
    # LEVEL 3 — Semi-urgent (8 rules)
    # ================================================================

    # R27: Moderate fever + symptoms
    if (temp is not None and 38.5 <= temp < 40.0
            and (sob_sev >= 1 or cp_sev >= 1 or pain >= 7)):
        reasons_l3.append(f"Fievre moderee (T={temp}) + symptomes")

    # R28: Mild tachycardia + symptoms
    if (hr is not None and 100 < hr <= 140
            and (sob_sev >= 1 or cp_sev >= 1)):
        reasons_l3.append(f"Tachycardie legere (FC={hr}) + symptomes respi/DT")

    # R29: Severe pain with physiological impact
    if pain >= 8 and hr is not None and hr > 100:
        reasons_l3.append(f"Douleur severe (EVA={pain}) + retentissement (FC={hr})")

    # R30: SpO2 92-94
    if spo2 is not None and 92 <= spo2 <= 94:
        reasons_l3.append(f"SpO2={spo2}% (limite basse)")

    # R31: Elderly + abnormal vital
    if age >= 70 and (
        (hr is not None and (hr > 100 or hr < 55))
        or (sbp is not None and sbp < 110)
        or (temp is not None and temp >= 38.0)
    ):
        reasons_l3.append(f"Patient age ({age} ans) + parametre vital anormal")

    # R32: Multi-comorbidity (>= 2) with any symptom
    if num_comorbidities >= 2 and (sob_sev >= 1 or cp_sev >= 1 or pain >= 6):
        reasons_l3.append(f"Multi-comorbidite ({num_comorbidities}) + symptomes")

    # R33: Exertional dyspnea + respiratory history
    if sob_sev >= 1 and respiratory == 1 and spo2 is not None and spo2 <= 96:
        reasons_l3.append("Dyspnee + ATCD respiratoire + SpO2 limite")

    # R34: Mild chest pain + cardiac patient aged 50+
    if cp_sev >= 1 and cardiac == 1 and age >= 50:
        reasons_l3.append("DT legere + patient cardiaque age")

    if reasons_l3:
        return 3, reasons_l3

    return None, []


def inconsistency_score(v):
    """
    Symptom vs vitals inconsistency heuristic (6 patterns).
    Higher score = more suspicious -> lower confidence.
    """
    pain = v.get("pain", 0)
    hr = v.get("hr", 80)
    rr = v.get("resp_rate", 16)
    spo2 = v.get("spo2", 98)
    sbp = v.get("sbp", 120)
    cp_sev = v.get("chest_pain_severity", 0)
    sob_sev = v.get("sob_severity", 0)
    confusion = v.get("confusion", 0)

    # Backward compat
    if cp_sev == 0 and v.get("chest_pain", 0) == 1:
        cp_sev = 2
    if sob_sev == 0 and v.get("sob", 0) == 1:
        sob_sev = 2

    score = 0.0

    # P1: Extreme pain but perfectly normal vitals and no cardio/resp complaint
    if pain >= 9 and hr < 90 and rr < 18 and spo2 >= 97 and cp_sev == 0 and sob_sev == 0:
        score += 1.0

    # P2: "I can't breathe" but objective looks fine
    if sob_sev >= 2 and rr < 18 and spo2 >= 97:
        score += 0.8

    # P3: Chest pain but vitals entirely normal
    if cp_sev >= 2 and hr < 90 and rr < 18 and spo2 >= 97 and sbp >= 110:
        score += 0.4

    # P4: Confusion reported but vitals perfectly stable
    if confusion == 1 and hr < 90 and sbp >= 110 and spo2 >= 97 and rr < 20:
        score += 0.6

    # P5: No symptoms at all but abnormal vitals
    if (pain <= 2 and cp_sev == 0 and sob_sev == 0 and confusion == 0
            and (hr > 120 or rr > 28 or spo2 < 92)):
        score += 0.5

    # P6: Multiple severe symptoms but perfect vitals
    severe_count = (cp_sev >= 2) + (sob_sev >= 2) + (pain >= 8) + (confusion == 1)
    if severe_count >= 2 and hr < 85 and rr < 16 and spo2 >= 98:
        score += 0.9

    return score
