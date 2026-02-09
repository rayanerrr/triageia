def rule_based_triage(v):
    """
    Educational rule engine.
    Returns (level, reasons) or (None, []) if no override.
    level: 1=critical, 2=urgent, 3=semi-urgent, 4=less urgent, 5=non-urgent
    """
    spo2 = v.get("spo2")
    rr = v.get("resp_rate")
    sbp = v.get("sbp")
    hr = v.get("hr")
    temp = v.get("temp")
    confusion = v.get("confusion", 0)

    # Hard safety overrides (objective-first)
    if confusion == 1:
        return 1, ["Confusion/altération de l’état mental"]

    if spo2 is not None and spo2 < 90:
        return 1, ["SpO2<90 (hypoxie sévère)"]

    if sbp is not None and sbp < 90:
        return 1, ["SBP<90 (hypotension/choc)"]

    if rr is not None and rr > 30:
        return 2, ["FR>30 (détresse respiratoire)"]

    if hr is not None and hr > 140:
        return 2, ["FC>140 (tachycardie sévère)"]

    if temp is not None and temp >= 40.0:
        return 2, ["Temp>=40.0 (hyperthermie)"]

    return None, []


def inconsistency_score(v):
    """
    Simple 'symptom vs vitals' inconsistency heuristic.
    Higher = more suspicious -> slightly lower confidence.
    """
    pain = v.get("pain", 0)
    hr = v.get("hr", 80)
    rr = v.get("resp_rate", 16)
    spo2 = v.get("spo2", 98)
    chest_pain = v.get("chest_pain", 0)
    sob = v.get("sob", 0)

    score = 0.0

    # Extreme pain but normal vitals and no cardio/resp complaint
    if pain >= 9 and hr < 90 and rr < 18 and spo2 >= 96 and chest_pain == 0 and sob == 0:
        score += 1.0

    # "I can't breathe" but objective looks fine
    if sob == 1 and rr < 18 and spo2 >= 96:
        score += 0.7

    # Chest pain but vitals normal (still could be real; small penalty)
    if chest_pain == 1 and hr < 95 and rr < 20 and spo2 >= 95:
        score += 0.3

    return score