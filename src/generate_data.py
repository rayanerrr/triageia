import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Realistic triage dataset generator (50 000 patients).
# Output: data/triage.csv
# Label: triage_level in {1..5} (1=critical, 5=non-urgent)
#
# Features:
#   age, hr, sbp, dbp, resp_rate, temp, spo2, pain,
#   chest_pain_severity (0-3), sob_severity (0-3), confusion,
#   comorbidity_cardiac, comorbidity_respiratory,
#   comorbidity_diabetes, num_comorbidities
# ------------------------------------------------------------

def _clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


def _bimodal_ages(rng, n):
    """25% pediatric (peak ~5), 75% adult (peak ~55)."""
    n_ped = int(n * 0.25)
    n_adult = n - n_ped
    ped = rng.normal(5, 4, size=n_ped)
    ped = np.clip(ped, 0, 17).astype(int)
    adult = rng.normal(55, 18, size=n_adult)
    adult = np.clip(adult, 18, 100).astype(int)
    ages = np.concatenate([ped, adult])
    rng.shuffle(ages)
    return ages


def _pediatric_vitals(rng, ages, hr, rr, sbp):
    """Adjust vitals for pediatric patients (age-dependent normals)."""
    for i in range(len(ages)):
        age = ages[i]
        if age <= 1:  # nourrisson
            hr[i] = rng.normal(130, 15)
            rr[i] = rng.normal(40, 8)
            sbp[i] = rng.normal(75, 10)
        elif age <= 5:
            hr[i] = rng.normal(110, 12)
            rr[i] = rng.normal(28, 5)
            sbp[i] = rng.normal(90, 10)
        elif age <= 12:
            hr[i] = rng.normal(95, 10)
            rr[i] = rng.normal(22, 4)
            sbp[i] = rng.normal(100, 10)
        elif age <= 17:
            hr[i] = rng.normal(82, 10)
            rr[i] = rng.normal(18, 3)
            sbp[i] = rng.normal(112, 10)
    return hr, rr, sbp


def _generate_comorbidities(rng, ages, n):
    """Age-dependent comorbidity generation."""
    p_cardiac = np.where(ages >= 60, 0.25, np.where(ages >= 40, 0.10, 0.02))
    p_resp = np.where(ages >= 50, 0.15, np.where(ages <= 12, 0.12, 0.08))
    p_diab = np.where(ages >= 50, 0.20, np.where(ages >= 30, 0.08, 0.01))

    cardiac = rng.binomial(1, p_cardiac, size=n)
    respiratory = rng.binomial(1, p_resp, size=n)
    diabetes = rng.binomial(1, p_diab, size=n)
    num = cardiac + respiratory + diabetes
    return cardiac, respiratory, diabetes, num


def triage_label_rule(age, hr, sbp, rr, temp, spo2, pain,
                      chest_pain_sev, sob_sev, confusion,
                      cardiac, respiratory, diabetes, num_comorbidities,
                      rng):
    """
    Nuanced labeling with stochastic noise to avoid auto-reference
    with the rule engine. Returns triage level 1..5.
    """
    score = 0.0

    # --- Critical objective signs ---
    if spo2 < 88:
        score += 8
    elif spo2 < 90:
        score += 6
    elif spo2 < 92:
        score += 4
    elif spo2 < 95:
        score += 2

    if rr > 35:
        score += 6
    elif rr > 28:
        score += 4
    elif rr > 22:
        score += 1.5

    if rr < 6:
        score += 7
    elif rr < 8:
        score += 4

    if sbp < 80:
        score += 7
    elif sbp < 90:
        score += 5
    elif sbp < 100:
        score += 2

    if hr > 150:
        score += 5
    elif hr > 130:
        score += 3.5
    elif hr > 110:
        score += 2
    elif hr > 100:
        score += 0.8

    if hr < 35:
        score += 6
    elif hr < 45:
        score += 3.5
    elif hr < 50:
        score += 1.5

    if temp >= 41.0:
        score += 5
    elif temp >= 40.0:
        score += 3.5
    elif temp >= 39.0:
        score += 2
    elif temp >= 38.5:
        score += 1

    if temp <= 33.5:
        score += 5
    elif temp <= 35.0:
        score += 2

    # --- Symptoms ---
    if confusion:
        score += 6

    # sob_severity 0-3
    score += sob_sev * 2.2

    # chest_pain_severity 0-3
    score += chest_pain_sev * 1.5

    # Pain (subjective, less weight)
    if pain >= 9:
        score += 1.2
    elif pain >= 7:
        score += 0.6
    elif pain >= 5:
        score += 0.3

    # --- Demographics ---
    if age >= 75:
        score += 1.8
    elif age >= 65:
        score += 1.0
    elif age <= 1:
        score += 1.5  # nourrissons more vulnerable

    # --- Comorbidities ---
    if cardiac and chest_pain_sev >= 1:
        score += 2.5
    elif cardiac:
        score += 0.8

    if respiratory and sob_sev >= 1:
        score += 2.0
    elif respiratory:
        score += 0.5

    if diabetes:
        score += 0.4

    if num_comorbidities >= 3:
        score += 1.5
    elif num_comorbidities >= 2:
        score += 0.7

    # --- Interactions ---
    # Septic triad
    if temp >= 38.5 and hr > 100 and rr > 22:
        score += 3

    # Chest pain + cardiac history + instability
    if chest_pain_sev >= 2 and cardiac and (hr > 110 or sbp < 100):
        score += 3

    # Base score from age (everyone gets a baseline)
    if age >= 50:
        score += 1.0
    elif age >= 30:
        score += 0.5

    # Add gaussian noise to avoid deterministic mapping
    noise = rng.normal(0, 2.0)
    score += noise

    # Map to levels (thresholds tuned for target distribution)
    if score >= 19:
        return 1
    if score >= 11:
        return 2
    if score >= 5.5:
        return 3
    if score >= 2.0:
        return 4
    return 5


def generate(n=50000, seed=42):
    rng = np.random.default_rng(seed)

    ages = _bimodal_ages(rng, n)

    # Base vitals (adult defaults, adjusted for pediatric below)
    hr = rng.normal(80, 15, size=n).astype(float)
    sbp = rng.normal(125, 18, size=n).astype(float)
    rr = rng.normal(16, 3, size=n).astype(float)
    temp = rng.normal(37.0, 0.5, size=n).astype(float)
    spo2 = rng.normal(97, 1.5, size=n).astype(float)

    # Adjust pediatric vitals
    hr, rr, sbp = _pediatric_vitals(rng, ages, hr, rr, sbp)

    # Comorbidities
    cardiac, respiratory, diabetes, num_comorbidities = _generate_comorbidities(rng, ages, n)

    # Symptoms
    # chest_pain_severity 0-3 (0=none, 1=mild, 2=moderate, 3=severe)
    p_cp = np.where(cardiac == 1, 0.20, 0.08)
    has_cp = rng.binomial(1, p_cp, size=n)
    chest_pain_sev = np.zeros(n, dtype=int)
    for i in range(n):
        if has_cp[i]:
            chest_pain_sev[i] = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    # sob_severity 0-3
    p_sob = np.where(respiratory == 1, 0.25, 0.10)
    has_sob = rng.binomial(1, p_sob, size=n)
    sob_sev = np.zeros(n, dtype=int)
    for i in range(n):
        if has_sob[i]:
            sob_sev[i] = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    # Confusion
    p_conf = np.where(ages >= 75, 0.08, np.where(ages >= 65, 0.05, 0.02))
    confusion = rng.binomial(1, p_conf, size=n)

    # Pain 0..10 correlated with symptoms
    pain = rng.normal(3, 2.5, size=n) + 2.0 * chest_pain_sev + 1.5 * sob_sev
    pain = np.clip(np.round(pain), 0, 10).astype(int)

    # Correlate vitals with symptoms (stochastic, with noise)
    for i in range(n):
        noise_hr = rng.normal(0, 5)
        noise_rr = rng.normal(0, 2)
        noise_sbp = rng.normal(0, 5)
        noise_spo2 = rng.normal(0, 1)

        hr[i] += (12 * sob_sev[i] + 8 * chest_pain_sev[i]
                   + 20 * confusion[i] + noise_hr)
        rr[i] += (6 * sob_sev[i] + 3 * confusion[i] + noise_rr)
        sbp[i] -= (15 * confusion[i] + 5 * sob_sev[i] + noise_sbp)
        spo2[i] -= (3 * sob_sev[i] + 4 * confusion[i] + noise_spo2)

    # Fever episodes (~10%)
    fever_mask = rng.random(n) < 0.10
    temp[fever_mask] += rng.normal(1.8, 0.6, size=fever_mask.sum())

    # Extreme/critical cases (~3%)
    extreme = rng.random(n) < 0.03
    n_extreme = extreme.sum()
    spo2[extreme] -= rng.uniform(8, 18, size=n_extreme)
    sbp[extreme] -= rng.uniform(20, 50, size=n_extreme)
    rr[extreme] += rng.uniform(8, 20, size=n_extreme)
    hr[extreme] += rng.uniform(20, 55, size=n_extreme)
    confusion[extreme] = 1

    # Bradycardia cases (~2%)
    brady = rng.random(n) < 0.02
    hr[brady] = rng.uniform(28, 48, size=brady.sum())

    # Hypothermia cases (~1%)
    hypo = rng.random(n) < 0.01
    temp[hypo] = rng.uniform(32.0, 35.0, size=hypo.sum())

    # Respiratory depression (~1%)
    resp_dep = rng.random(n) < 0.01
    rr[resp_dep] = rng.uniform(4, 8, size=resp_dep.sum())

    # Clamp to plausible ranges
    hr = _clamp(np.round(hr), 25, 220).astype(int)
    sbp = _clamp(np.round(sbp), 55, 240).astype(int)
    rr = _clamp(np.round(rr), 4, 60).astype(int)
    temp = _clamp(np.round(temp, 1), 32.0, 42.0)
    spo2 = _clamp(np.round(spo2), 60, 100).astype(int)

    # DBP derived
    dbp = sbp - rng.normal(42, 10, size=n)
    dbp = _clamp(np.round(dbp), 30, 150).astype(int)

    # Generate labels
    labels = np.array([
        triage_label_rule(
            int(ages[i]), float(hr[i]), float(sbp[i]), float(rr[i]),
            float(temp[i]), float(spo2[i]), int(pain[i]),
            int(chest_pain_sev[i]), int(sob_sev[i]), int(confusion[i]),
            int(cardiac[i]), int(respiratory[i]), int(diabetes[i]),
            int(num_comorbidities[i]), rng
        )
        for i in range(n)
    ], dtype=int)

    df = pd.DataFrame({
        "age": ages,
        "hr": hr,
        "sbp": sbp,
        "dbp": dbp,
        "resp_rate": rr,
        "temp": temp,
        "spo2": spo2,
        "pain": pain,
        "chest_pain_severity": chest_pain_sev,
        "sob_severity": sob_sev,
        "confusion": confusion,
        "comorbidity_cardiac": cardiac,
        "comorbidity_respiratory": respiratory,
        "comorbidity_diabetes": diabetes,
        "num_comorbidities": num_comorbidities,
        "triage_level": labels,
    })

    return df


if __name__ == "__main__":
    import os
    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    data_dir = os.path.join(_base, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate()
    out = os.path.join(data_dir, "triage.csv")
    df.to_csv(out, index=False)
    print(f"Saved {out} with shape={df.shape}")
    print("\nLabel distribution:")
    dist = df["triage_level"].value_counts().sort_index()
    print(dist)
    print("\nPercentages:")
    print((dist / len(df) * 100).round(1))
    print(f"\nAge range: {df['age'].min()} - {df['age'].max()}")
    print(f"Pediatric (<18): {(df['age'] < 18).sum()} ({(df['age'] < 18).mean()*100:.1f}%)")
    print(f"Comorbidities: cardiac={df['comorbidity_cardiac'].mean()*100:.1f}%, "
          f"resp={df['comorbidity_respiratory'].mean()*100:.1f}%, "
          f"diab={df['comorbidity_diabetes'].mean()*100:.1f}%")
