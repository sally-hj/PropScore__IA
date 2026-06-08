"""Project constants and feature definitions."""

from __future__ import annotations

LABELS = ["cold", "warm", "hot"]
LABEL_TO_ID = {"cold": 0, "warm": 1, "hot": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

TABULAR_RAW_LABEL_MAP = {
    "froid": "cold",
    "cold": "cold",
    "tiede": "warm",
    "tiède": "warm",
    "warm": "warm",
    "chaud": "hot",
    "hot": "hot",
}

TABULAR_RENAME_MAP = {
    "Age": "age",
    "Salaire_Mensuel": "income",
    "Profession": "profession",
    "Type_de_Bien": "property_type",
    "Budget_Cible": "budget",
    "Nb_Visites_Site": "num_visits",
    "Source_Lead": "source_lead",
    "Target": "binary_target",
    "Total Time Spent on Website": "total_time_spent_website",
    "Page Views Per Visit": "page_views_per_visit",
    "City": "city",
    "Lead Origin": "lead_origin",
    "Last Activity": "last_activity",
    "Last Notable Activity": "last_notable_activity",
    "Score_Lead": "score_lead",
    "Target_Multiclasse": "label_raw",
    "Ratio_Budget_Salaire": "ratio_budget_income",
    "Score_Engagement": "interaction_score",
}

TABULAR_NUMERIC_FEATURES = [
    "age",
    "income",
    "budget",
    "num_visits",
    "total_time_spent_website",
    "page_views_per_visit",
    "ratio_budget_income",
    "interaction_score",
]

TABULAR_CATEGORICAL_FEATURES = [
    "profession",
    "property_type",
    "source_lead",
    "city",
    "lead_origin",
    "last_activity",
    "last_notable_activity",
]

TABULAR_TARGET_COLUMNS = [
    "label_raw",
    "label",
    "target_multiclasse_id",
    "binary_target",
]

TABULAR_EXCLUDED_FEATURES = set(TABULAR_TARGET_COLUMNS + ["score_lead"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

EXPECTED_TABULAR_COLUMNS = [
    "prospect_id",
    "age",
    "income",
    "profession",
    "property_type",
    "budget",
    "num_visits",
    "source_lead",
    "binary_target",
    "total_time_spent_website",
    "page_views_per_visit",
    "city",
    "lead_origin",
    "last_activity",
    "last_notable_activity",
    "score_lead",
    "label_raw",
    "ratio_budget_income",
    "interaction_score",
    "label",
    "target_multiclasse_id",
]

NUMERIC_TABULAR_COLUMNS = [
    "age",
    "income",
    "budget",
    "num_visits",
    "total_time_spent_website",
    "page_views_per_visit",
    "score_lead",
    "ratio_budget_income",
    "interaction_score",
]

CATEGORICAL_TABULAR_COLUMNS = [
    "profession",
    "property_type",
    "source_lead",
    "city",
    "lead_origin",
    "last_activity",
    "last_notable_activity",
    "binary_target",
    "label_raw",
]

POSITIVE_KEYWORDS = [
    "interessé",
    "intéressée",
    "intéressé par",
    "ça m'intéresse",
    "je veux",
    "je suis prêt",
    "prêt",
    "achat",
    "acheter",
    "visite",
    "visiter",
    "réserver",
    "signer",
    "budget ok",
    "confirmer",
    "dossier",
    "financement",
    "oui",
    "yes",
    "bghit",
    "بغيت",
    "مهم",
    "villa",
    "appartement",
    "terrain",
]

HESITATION_KEYWORDS = [
    "peut-être",
    "peut etre",
    "je réfléchis",
    "réfléchir",
    "je vais voir",
    "on verra",
    "pas sûr",
    "pas sure",
    "à voir",
    "plus tard",
    "selon",
    "comparer",
    "j'attends",
    "j attends",
    "éventuellement",
    "hésite",
    "hésitation",
    "peut être",
]

NEGATIVE_KEYWORDS = [
    "non",
    "pas intéressé",
    "pas interesse",
    "refus",
    "refuser",
    "jamais",
    "sans suite",
    "arrêter",
    "arreter",
    "annuler",
    "pas maintenant",
    "trop cher",
    "inutile",
    "aucun intérêt",
    "aucun interet",
    "je ne veux pas",
    "pas de suite",
    "décliner",
    "disponibilité limitée",
    "pas concerné",
]

DEFAULT_WHISPER_MODEL = "base"
DEFAULT_CAMEMBERT_MODEL = "camembert-base"
