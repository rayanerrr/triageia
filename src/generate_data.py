import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Synthetic triage dataset generator (educational).
# Output: data/triage.csv
# Label: triage_level in {1..5} (1=critical, 5=non-urgent)
# ------------------------------------------------------------

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def triage_label_rule(age, hr, sbp, rr, temp, spo2, pain, chest_pain, sob, confusion):
    """
    Heuristic label generator (NOT clinical truth).
    Produces label 1..5 using a risk-like scoring logic.
    """
    score = 0.0

    # Objective vitals (strong weight)
    if spo2 < 90: score += 6
    elif spo2 < 92: score += 4
    elif spo2 < 95: score += 2

    if rr > 30: score += 5
    elif rr > 24: score += 3
    elif rr > 20: score += 1

    if sbp < 90: score += 6
    elif sbp < 100: score += 3

    if hr > 140: score += 4
    elif hr > 120: score += 3
    elif hr > 100: score += 1
    elif hr < 45: score += 3

    if temp >= 40.0: score += 4
    elif temp >= 39.0: score += 3
    elif temp >= 38.0: score += 1
    elif temp <= 35.0: score += 3

    # Demographics / context (medium)
    if age >= 75: score += 2
    elif age >= 65: score += 1

    # Symptoms (moderate)
    if confusion: score += 5
    if sob: score += 3
    if chest_pain: score += 2

    # Pain is subjective -> small contribution
    if pain >= 9: score += 1.0
    elif pain >= 7: score += 0.6
    elif pain >= 4: score += 0.2

    # Convert score to triage levels 1..5
    if score >= 12:
        return 1
    if score >= 8:
        return 2
    if score >= 5:
        return 3
    if score >= 2:
        return 4
    return 5

def generate(n=12000, seed=42):
    rng = np.random.default_rng(seed)

    ages = rng.integers(0, 95, size=n)

    # Base vitals
    hr = rng.normal(85, 18, size=n)
    sbp = rng.normal(120, 18, size=n)
    rr = rng.normal(16, 4, size=n)
    temp = rng.normal(37.0, 0.6, size=n)
    spo2 = rng.normal(97, 2, size=n)

    # Symptoms (binary)
    chest_pain = rng.binomial(1, 0.10, size=n)
    sob = rng.binomial(1, 0.12, size=n)        # shortness of breath
    confusion = rng.binomial(1, 0.04, size=n)

    # Pain 0..10 correlated with chest_pain / sob
    pain = rng.normal(3.5, 2.5, size=n) + 2.0 * chest_pain + 1.2 * sob
    pain = np.clip(np.round(pain), 0, 10)

    # Correlate physiological changes with symptoms
    hr += 18 * sob + 12 * chest_pain + 25 * confusion
    rr += 10 * sob + 6 * confusion
    sbp -= 22 * confusion + 8 * sob
    spo2 -= 10 * sob + 6 * confusion
    temp += 0.8 * (rng.random(n) < 0.08)  # occasional fever bump

    # Extreme cases (small fraction)
    extreme = rng.random(n) < 0.02
    spo2[extreme] -= rng.uniform(8, 20, size=extreme.sum())
    sbp[extreme] -= rng.uniform(25, 50, size=extreme.sum())
    rr[extreme] += rng.uniform(10, 25, size=extreme.sum())
    hr[extreme] += rng.uniform(20, 60, size=extreme.sum())
    confusion[extreme] = 1

    # Clamp to plausible ranges
    hr = np.array([_clamp(x, 30, 200) for x in hr])
    sbp = np.array([_clamp(x, 60, 220) for x in sbp])
    rr = np.array([_clamp(x, 6, 60) for x in rr])
    temp = np.array([_clamp(x, 33.0, 41.5) for x in temp])
    spo2 = np.array([_clamp(x, 70, 100) for x in spo2])

    # DBP derived (rough)
    dbp = sbp - rng.normal(45, 10, size=n)
    dbp = np.array([_clamp(x, 35, 140) for x in dbp])

    labels = [
        triage_label_rule(
            int(ages[i]),
            float(hr[i]),
            float(sbp[i]),
            float(rr[i]),
            float(temp[i]),
            float(spo2[i]),
            int(pain[i]),
            int(chest_pain[i]),
            int(sob[i]),
            int(confusion[i]),
        )
        for i in range(n)
    ]

    df = pd.DataFrame({
        "age": ages,
        "hr": np.round(hr).astype(int),
        "sbp": np.round(sbp).astype(int),
        "dbp": np.round(dbp).astype(int),
        "resp_rate": np.round(rr).astype(int),
        "temp": np.round(temp, 1),
        "spo2": np.round(spo2).astype(int),
        "pain": pain.astype(int),
        "chest_pain": chest_pain.astype(int),
        "sob": sob.astype(int),
        "confusion": confusion.astype(int),
        "triage_level": np.array(labels, dtype=int),
    })

    return df

if __name__ == "__main__":
    df = generate()
    out = "data/triage.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} with shape={df.shape}")
    print("Label distribution:")
    print(df["triage_level"].value_counts().sort_index())