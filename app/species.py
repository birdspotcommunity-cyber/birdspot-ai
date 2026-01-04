import json
import os

SPECIES_CACHE = None

def load_species():
    global SPECIES_CACHE
    if SPECIES_CACHE is not None:
        return SPECIES_CACHE

    species_file = os.getenv("SPECIES_FILE", "./data/species_list.json")
    with open(species_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # expected format: [{"id":"...", "species_name":"...", "scientific_name":"..."}]
    SPECIES_CACHE = data
    return SPECIES_CACHE

def match_species(pred_name: str, pred_sci: str):
    species = load_species()

    pred_name_l = (pred_name or "").strip().lower()
    pred_sci_l = (pred_sci or "").strip().lower()

    # match by scientific name first
    if pred_sci_l:
        for s in species:
            if (s.get("scientific_name","").strip().lower() == pred_sci_l):
                return s

    # then common name
    if pred_name_l:
        for s in species:
            if s.get("species_name","").strip().lower() == pred_name_l:
                return s

    return None
