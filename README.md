# POC Supplier Entity Resolution

## Purpose
This project demonstrates a Proof of Concept for **entity resolution** of companies in a client’s database. The goal is to correctly identify and match suppliers, remove duplicates, and prepare a clean dataset for analysis.

## What the code does
1. Loads a file with companies and possible candidate matches.
2. Normalizes company names and removes stopwords.
3. Calculates matching scores based on:
   - Name score (72%): receives the highest weight because, in general, the similarity of the company name is the strongest matching signal
   - Country score (18%): has a lower weight, because companies can sometimes appear in multiple countries (subsidiaries, branches)
   - Website score (10%): has a small role, but helps with discrimination (a company with a website is an additional indicator of legitimacy and differentiation).
   - Threshold 0.55: means that a candidate is accepted as a match only if the final score is at least 55%.
4. Selects the best candidate for each input row.
5. Marks each row as `matched` or `unmatched`. This method detects duplicates (the same company written in different ways) and measures the correlation between the input and each candidate.
6. Generates a summary report file with the results.

## How to run
Make sure Python and `pandas` are installed.

## Python code
"""
POC Entity Resolution - script Python
-------------------------------------

Scop: simulare proces de curățare și potrivire a entităților (companii) pe baza unui CSV de input + candidate.
Limbă: comentarii și explicații în română.
"""
from typing import List

import pandas as pd #pentru lucrul cu tabele (DataFrame)
import re #expresii regulate, folosite pentru curățarea textului
from difflib import SequenceMatcher #compară două șiruri și calculează o măsură de similaritate
from pathlib import Path #pentru a lucra cu foldere și fișiere într-un mod portabil

# -----------------------------
# 1. Încărcare dataset
# -----------------------------
df = pd.read_csv("C:\\Users\\Cristi\\Desktop\\presales_data_sample.csv")

# -----------------------------
# 2. Funcții utilitare
# -----------------------------
COMPANY_STOPWORDS = [
    "limited", "private", "ltd", "inc", "srl", "sa", "gmbh", "llc", "co", "corp",
    "company", "corporation", "pte", "plc", "nv", "oy", "ag", "spa", "ab", "bv",
    "kk", "kft", "ltda", "pvt", "sas", "bvba"
]     #Listează cuvintele comune care apar în denumirile companiilor (Ltd, Inc, Srl etc.)
      #Ele vor fi eliminate în procesul de normalizare, pentru a evita potrivirea greșită pe baza acestor cuvinte

def normalize_name(s: str) -> str:
    """Normalizează un nume de companie: lowercase, elimină semne, stopwords și spații multiple"""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace("ș", "s").replace("ț", "t").replace("ă", "a").replace("î", "i").replace("â", "a")
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [t for t in s.split() if t and t not in COMPANY_STOPWORDS]
    return " ".join(tokens)    #Transformă numele în lowercase, elimină diacriticele și caracterele speciale
                               #Împarte textul în token-uri și elimină stopwords
                               #Exemplu: "ACME Ltd. România" -> "acme romania"

def seq_ratio(a: str, b: str) -> float:
    """Similaritate secvențială (difflib)"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()  #Măsoară cât de similare sunt două șiruri de text (valoare între 0 și 1)

def token_set_ratio(a: str, b: str) -> float:
    """Simplified token set ratio: măsoară proporția de tokenuri comune"""
    toks_a = set(a.split())
    toks_b = set(b.split())
    if not toks_a or not toks_b:
        return 0.0
    inter = toks_a.intersection(toks_b)
    return (2 * len(inter)) / (len(toks_a) + len(toks_b))  #Compară token-urile din cele două șiruri și calculează proporția de tokenuri comune
                                                           #Este util pentru a detecta potriviri chiar dacă ordinea cuvintelor diferă

# -----------------------------
# 3. Pre-procesare dataset
# -----------------------------
df["_norm_input_name"] = df["input_company_name"].apply(normalize_name)
df["_norm_company_name"] = df["company_name"].apply(normalize_name)
df["_norm_legal_names"] = df["company_legal_names"].fillna("").apply(normalize_name)
df["_norm_commercial_names"] = df["company_commercial_names"].fillna("").apply(normalize_name)
df["_candidate_website_present"] = df["website_url"].notna()   #Creează coloane noi cu numele normalizate pentru toate câmpurile relevante
                                                               #Creează o coloană booleană care arată dacă există un website pentru fiecare candidat

# -----------------------------
# 4. Funcție de scor pentru fiecare candidat
# -----------------------------
def score_candidate(row):
    input_norm = row["_norm_input_name"]

    name_scores = [seq_ratio(input_norm, row["_norm_company_name"])]
    if row["_norm_legal_names"]:
        name_scores.append(seq_ratio(input_norm, row["_norm_legal_names"]))
        name_scores.append(token_set_ratio(input_norm, row["_norm_legal_names"]))
    if row["_norm_commercial_names"]:
        name_scores.append(seq_ratio(input_norm, row["_norm_commercial_names"]))
        name_scores.append(token_set_ratio(input_norm, row["_norm_commercial_names"]))

    name_score = max(name_scores) if name_scores else 0.0

    # scor țară
    country_score = 0.0
    if pd.notna(row["input_main_country_code"]) and pd.notna(row["main_country_code"]):
        if str(row["input_main_country_code"]).strip().upper() == str(row["main_country_code"]).strip().upper():
            country_score = 1.0
    elif pd.notna(row["input_main_country"]) and pd.notna(row["main_country"]):
        if str(row["input_main_country"]).strip().lower() == str(row["main_country"]).strip().lower():
            country_score = 0.9

    website_score = 1.0 if row["_candidate_website_present"] else 0.0

    final = 0.72 * name_score + 0.18 * country_score + 0.10 * website_score

    return {
        "name_score": round(name_score, 4),
        "country_score": round(country_score, 4),
        "website_score": int(website_score),
        "final_score": round(final, 4),
    }

#Pentru fiecare candidat:
#Calculează scorul numelui folosind seq_ratio și token_set_ratio.
#Compară țara input cu țara candidatului (1.0 dacă exact match, 0.9 dacă match pe nume țară).
#Verifică dacă website-ul există (1.0 sau 0.0).
#Combină scorurile cu ponderi:
#Name: 0.72
#Country: 0.18
#Website: 0.10
#Returnează un dicționar cu scorurile individuale și scorul final.


# -----------------------------
# 5. Alegerea celui mai bun candidat per input_row_key
# -----------------------------
rows_out = []
grouped = df.groupby("input_row_key")

for key, group in grouped:
    cand_scores = []
    for idx, row in group.iterrows():
        sc = score_candidate(row)
        cand_scores.append((idx, sc["final_score"], sc, row))

    cand_scores_sorted = sorted(cand_scores, key=lambda x: x[1], reverse=True)
    best_idx, best_score, best_scoredict, best_row = cand_scores_sorted[0]

    threshold = 0.55
    decision = "matched" if best_score >= threshold else "unmatched"


#Grupează rândurile după input_row_key (fiecare companie de input).
#Pentru fiecare grup:
#Calculează scorul final pentru toți candidații.
#Sortează descrescător și ia cel mai bun candidat.
#Aplică threshold = 0.55 pentru a decide dacă este matched sau unmatched.


    reason = []

    # verificare scor nume
    if best_scoredict["name_score"] >= 0.85:
        reason.append("name exact")
    elif best_scoredict["name_score"] >= 0.6:
        reason.append("name good")
    else:
        reason.append("name poor")

    # verificare țară
    if best_scoredict["country_score"] == 1:
        reason.append("country match")
    else:
        reason.append("country mismatch")

    # verificare website
    if best_scoredict["website_score"] == 1:
        reason.append("website present")
    else:
        reason.append("website missing")

    reason_short = "; ".join(reason)

    rows_out.append({
        "input_row_key": key,
        "input_company_name": best_row["input_company_name"],
        "best_candidate_idx": best_idx,
        "best_candidate_company_name": best_row["company_name"],
        "best_candidate_veridion_id": best_row.get("veridion_id", None),
        "best_candidate_main_country": best_row.get("main_country", None),
        "best_candidate_website": best_row.get("website_url", None),
        "name_score": best_scoredict["name_score"],
        "country_score": best_scoredict["country_score"],
        "website_score": best_scoredict["website_score"],
        "final_score": best_scoredict["final_score"],
        "decision": decision,
        "reason_short": reason_short,
    })

results = pd.DataFrame(rows_out)

#Creează un text scurt care explică decizia:
#Numele (exact, bun, slab)
#Țara (match/mismatch)
#Website (present/missing)


# -----------------------------
# 6. Salvare rezultate
# -----------------------------
out_dir = Path("presales_er_output")
out_dir.mkdir(parents=True, exist_ok=True)
# Creează un folder pentru rezultate dacă nu există.

# încercăm să scriem fișierul, dacă e deschis sau blocat, adăugăm sufix
out_file = out_dir / "er_matches.csv"
try:
    results.to_csv(out_file, index=False)
except PermissionError:
    out_file = out_dir / "er_matches_1.csv"
    results.to_csv(out_file, index=False)
# Salvează rezultatele într-un CSV.

# Raport sumar simplu
qc_summary = {
    "total_inputs": results.shape[0],
    "matched_count": int((results["decision"] == "matched").sum()),
    "unmatched_count": int((results["decision"] == "unmatched").sum()),
}

pd.DataFrame([qc_summary]).to_csv(out_dir / "er_qc_summary.csv", index=False)

#Creează un mic raport care arată:
#Numărul total de input-uri
#Câte rânduri au fost potrivite (matched)
#Câte au rămas nemarcate (unmatched)

print("Procesare completă!")
print("Rezultatele sunt salvate în folderul:", out_dir)
#Afișează mesaj de finalizare și folderul unde au fost salvate rezultatele.
"""
POC Entity Resolution - script Python
-------------------------------------

Scop: simulare proces de curățare și potrivire a entităților (companii) pe baza unui CSV de input + candidate.
Limbă: comentarii și explicații în română.
"""
from typing import List

import pandas as pd #pentru lucrul cu tabele (DataFrame)
import re #expresii regulate, folosite pentru curățarea textului
from difflib import SequenceMatcher #compară două șiruri și calculează o măsură de similaritate
from pathlib import Path #pentru a lucra cu foldere și fișiere într-un mod portabil

# -----------------------------
# 1. Încărcare dataset
# -----------------------------
df = pd.read_csv("C:\\Users\\Cristi\\Desktop\\presales_data_sample.csv")

# -----------------------------
# 2. Funcții utilitare
# -----------------------------
COMPANY_STOPWORDS = [
    "limited", "private", "ltd", "inc", "srl", "sa", "gmbh", "llc", "co", "corp",
    "company", "corporation", "pte", "plc", "nv", "oy", "ag", "spa", "ab", "bv",
    "kk", "kft", "ltda", "pvt", "sas", "bvba"
]     #Listează cuvintele comune care apar în denumirile companiilor (Ltd, Inc, Srl etc.)
      #Ele vor fi eliminate în procesul de normalizare, pentru a evita potrivirea greșită pe baza acestor cuvinte

def normalize_name(s: str) -> str:
    """Normalizează un nume de companie: lowercase, elimină semne, stopwords și spații multiple"""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace("ș", "s").replace("ț", "t").replace("ă", "a").replace("î", "i").replace("â", "a")
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [t for t in s.split() if t and t not in COMPANY_STOPWORDS]
    return " ".join(tokens)    #Transformă numele în lowercase, elimină diacriticele și caracterele speciale
                               #Împarte textul în token-uri și elimină stopwords
                               #Exemplu: "ACME Ltd. România" -> "acme romania"

def seq_ratio(a: str, b: str) -> float:
    """Similaritate secvențială (difflib)"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()  #Măsoară cât de similare sunt două șiruri de text (valoare între 0 și 1)

def token_set_ratio(a: str, b: str) -> float:
    """Simplified token set ratio: măsoară proporția de tokenuri comune"""
    toks_a = set(a.split())
    toks_b = set(b.split())
    if not toks_a or not toks_b:
        return 0.0
    inter = toks_a.intersection(toks_b)
    return (2 * len(inter)) / (len(toks_a) + len(toks_b))  #Compară token-urile din cele două șiruri și calculează proporția de tokenuri comune
                                                           #Este util pentru a detecta potriviri chiar dacă ordinea cuvintelor diferă

# -----------------------------
# 3. Pre-procesare dataset
# -----------------------------
df["_norm_input_name"] = df["input_company_name"].apply(normalize_name)
df["_norm_company_name"] = df["company_name"].apply(normalize_name)
df["_norm_legal_names"] = df["company_legal_names"].fillna("").apply(normalize_name)
df["_norm_commercial_names"] = df["company_commercial_names"].fillna("").apply(normalize_name)
df["_candidate_website_present"] = df["website_url"].notna()   #Creează coloane noi cu numele normalizate pentru toate câmpurile relevante
                                                               #Creează o coloană booleană care arată dacă există un website pentru fiecare candidat

# -----------------------------
# 4. Funcție de scor pentru fiecare candidat
# -----------------------------
def score_candidate(row):
    input_norm = row["_norm_input_name"]

    name_scores = [seq_ratio(input_norm, row["_norm_company_name"])]
    if row["_norm_legal_names"]:
        name_scores.append(seq_ratio(input_norm, row["_norm_legal_names"]))
        name_scores.append(token_set_ratio(input_norm, row["_norm_legal_names"]))
    if row["_norm_commercial_names"]:
        name_scores.append(seq_ratio(input_norm, row["_norm_commercial_names"]))
        name_scores.append(token_set_ratio(input_norm, row["_norm_commercial_names"]))

    name_score = max(name_scores) if name_scores else 0.0

    # scor țară
    country_score = 0.0
    if pd.notna(row["input_main_country_code"]) and pd.notna(row["main_country_code"]):
        if str(row["input_main_country_code"]).strip().upper() == str(row["main_country_code"]).strip().upper():
            country_score = 1.0
    elif pd.notna(row["input_main_country"]) and pd.notna(row["main_country"]):
        if str(row["input_main_country"]).strip().lower() == str(row["main_country"]).strip().lower():
            country_score = 0.9

    website_score = 1.0 if row["_candidate_website_present"] else 0.0

    final = 0.72 * name_score + 0.18 * country_score + 0.10 * website_score

    return {
        "name_score": round(name_score, 4),
        "country_score": round(country_score, 4),
        "website_score": int(website_score),
        "final_score": round(final, 4),
    }

#Pentru fiecare candidat:
#Calculează scorul numelui folosind seq_ratio și token_set_ratio.
#Compară țara input cu țara candidatului (1.0 dacă exact match, 0.9 dacă match pe nume țară).
#Verifică dacă website-ul există (1.0 sau 0.0).
#Combină scorurile cu ponderi:
#Name: 0.72
#Country: 0.18
#Website: 0.10
#Returnează un dicționar cu scorurile individuale și scorul final.


# -----------------------------
# 5. Alegerea celui mai bun candidat per input_row_key
# -----------------------------
rows_out = []
grouped = df.groupby("input_row_key")

for key, group in grouped:
    cand_scores = []
    for idx, row in group.iterrows():
        sc = score_candidate(row)
        cand_scores.append((idx, sc["final_score"], sc, row))

    cand_scores_sorted = sorted(cand_scores, key=lambda x: x[1], reverse=True)
    best_idx, best_score, best_scoredict, best_row = cand_scores_sorted[0]

    threshold = 0.55
    decision = "matched" if best_score >= threshold else "unmatched"


#Grupează rândurile după input_row_key (fiecare companie de input).
#Pentru fiecare grup:
#Calculează scorul final pentru toți candidații.
#Sortează descrescător și ia cel mai bun candidat.
#Aplică threshold = 0.55 pentru a decide dacă este matched sau unmatched.


    reason = []

    # verificare scor nume
    if best_scoredict["name_score"] >= 0.85:
        reason.append("name exact")
    elif best_scoredict["name_score"] >= 0.6:
        reason.append("name good")
    else:
        reason.append("name poor")

    # verificare țară
    if best_scoredict["country_score"] == 1:
        reason.append("country match")
    else:
        reason.append("country mismatch")

    # verificare website
    if best_scoredict["website_score"] == 1:
        reason.append("website present")
    else:
        reason.append("website missing")

    reason_short = "; ".join(reason)

    rows_out.append({
        "input_row_key": key,
        "input_company_name": best_row["input_company_name"],
        "best_candidate_idx": best_idx,
        "best_candidate_company_name": best_row["company_name"],
        "best_candidate_veridion_id": best_row.get("veridion_id", None),
        "best_candidate_main_country": best_row.get("main_country", None),
        "best_candidate_website": best_row.get("website_url", None),
        "name_score": best_scoredict["name_score"],
        "country_score": best_scoredict["country_score"],
        "website_score": best_scoredict["website_score"],
        "final_score": best_scoredict["final_score"],
        "decision": decision,
        "reason_short": reason_short,
    })

results = pd.DataFrame(rows_out)

#Creează un text scurt care explică decizia:
#Numele (exact, bun, slab)
#Țara (match/mismatch)
#Website (present/missing)


# -----------------------------
# 6. Salvare rezultate
# -----------------------------
out_dir = Path("presales_er_output")
out_dir.mkdir(parents=True, exist_ok=True)
# Creează un folder pentru rezultate dacă nu există.

# încercăm să scriem fișierul, dacă e deschis sau blocat, adăugăm sufix
out_file = out_dir / "er_matches.csv"
try:
    results.to_csv(out_file, index=False)
except PermissionError:
    out_file = out_dir / "er_matches_1.csv"
    results.to_csv(out_file, index=False)
# Salvează rezultatele într-un CSV.

# Raport sumar simplu
qc_summary = {
    "total_inputs": results.shape[0],
    "matched_count": int((results["decision"] == "matched").sum()),
    "unmatched_count": int((results["decision"] == "unmatched").sum()),
}

pd.DataFrame([qc_summary]).to_csv(out_dir / "er_qc_summary.csv", index=False)

#Creează un mic raport care arată:
#Numărul total de input-uri
#Câte rânduri au fost potrivite (matched)
#Câte au rămas nemarcate (unmatched)

print("Procesare completă!")
print("Rezultatele sunt salvate în folderul:", out_dir)
#Afișează mesaj de finalizare și folderul unde au fost salvate rezultatele.

