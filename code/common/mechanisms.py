"""Shared: physiological mechanism code inventory, directed edge definitions, message schema, and gating logic."""

# Allowed mechanism codes per directed edge, and the physiological carrier for each edge.
EDGE_MECHANISMS = {
    # Digestive system: stomach, intestine, gallbladder
    "stomach->intestine": {
        "carrier": "chyme (material flow)",
        "mechanisms": ["gastric_empty_release", "acid_flux", "osmotic_load", "nutrient_load"],
    },
    "intestine->stomach": {
        "carrier": "neural (vagovagal) + hormonal (CCK/secretin/GIP)",
        "mechanisms": ["entero_gastric_reflex", "cck_inhibit_empty", "secretin_inhibit_acid"],
    },
    "intestine->gallbladder": {
        "carrier": "hormonal (CCK) + vagal",
        "mechanisms": ["cck_gb_contract", "cck_sphincter_relax", "motilin_cycle"],
    },
    "gallbladder->intestine": {
        "carrier": "bile (material flow)",
        "mechanisms": ["bile_emulsify", "bile_alkaline_buffer", "bile_bile_salt", "bile_pressure_release"],
    },
    "gallbladder->stomach": {
        "carrier": "bile reflux (abnormal)",
        "mechanisms": ["bile_reflux_injury", "bile_reflux_pylorus", "bile_stasis_flowback"],
    },
    "stomach->gallbladder": {
        "carrier": "vagal + gastrin (weak)",
        "mechanisms": ["cephalic_gb_precontract", "gastrin_gb_tone", "stomach_distension_gb"],
    },
    # Pneumonia multi-organ (lung, heart, kidney, brain, liver, intestine): 14 directed edges
    "lung->heart": {
        "carrier": "hypoxia / acid-base / pressure",
        "mechanisms": ["hypoxia_increase_rv_load", "pulmonary_htn_rv_strain", "resp_acidosis_cv_depress"],
    },
    "heart->lung": {
        "carrier": "volume / pressure",
        "mechanisms": ["lv_failure_pulmonary_congestion", "volume_overload_lung_edema", "reduced_cop_lung_perfusion"],
    },
    "lung->kidney": {
        "carrier": "hypoxia / neurohumoral",
        "mechanisms": ["hypoxia_renal_vasoconstrict", "renal_hypoperfusion_raas", "hypoxia_juxtaglomerular_stimulate"],
    },
    "kidney->lung": {
        "carrier": "volume / acid-base / metabolism",
        "mechanisms": ["oliguria_fluid_retention_edema", "uremia_pulmonary_toxicity", "metabolic_acidosis_hypervent"],
    },
    "lung->brain": {
        "carrier": "hypoxia / carbon dioxide",
        "mechanisms": ["hypoxia_brain_dysfunction", "hypercapnia_cerebral_edema", "hypoxia_cerebral_ischemia"],
    },
    "brain->lung": {
        "carrier": "neural / respiratory drive",
        "mechanisms": ["depressed_drive_hypoventilation", "cns_injury_neurogenic_pulmonary_edema", "central_sleep_apnea_oxygen_desat"],
    },
    "lung->liver": {
        "carrier": "hypoxia / circulation",
        "mechanisms": ["hypoxia_hepatic_ischemia", "congestive_hepatic_vein", "hypoxia_hepatic_metabolism"],
    },
    "liver->lung": {
        "carrier": "metabolism / inflammation",
        "mechanisms": ["hepatic_ammonia_pulmonary", "liver_inflammation_cytokine", "portosystemic_shunt_hypoxia"],
    },
    "lung->intestine": {
        "carrier": "hypoxia / perfusion",
        "mechanisms": ["hypoxia_gut_barrier", "gut_ischemia_mucosal", "splanchnic_vasoconstrict"],
    },
    "intestine->lung": {
        "carrier": "inflammation / endotoxin",
        "mechanisms": ["gut_translocation_inflammation", "endotoxin_trigger_ards", "gut_cytokine_pulmonary"],
    },
    "heart->kidney": {
        "carrier": "cardiac output / volume",
        "mechanisms": ["low_cardiac_output_renal_hypoperfusion", "congestion_renal_venous", "raas_activation_sodium_retain"],
    },
    "kidney->heart": {
        "carrier": "volume / electrolytes",
        "mechanisms": ["fluid_retention_heart_load", "hyperkalemia_cardiotoxicity", "uremia_myocardium"],
    },
    "heart->brain": {
        "carrier": "perfusion / embolus",
        "mechanisms": ["embolism_cerebral_infarct", "low_perfusion_cerebral", "hypotension_cerebral_ischemia"],
    },
    "brain->heart": {
        "carrier": "neural / stress",
        "mechanisms": ["cns_injury_neurohormonal_stress", "stress_catecholamine_cardiac", "autonomic_dysregulation_arrhythmia"],
    },
    "kidney->liver": {
        "carrier": "metabolism / volume",
        "mechanisms": ["hepatorenal_circulation", "uremia_hepatic_metabolism", "acid_base_hepatic"],
    },
    "liver->kidney": {
        "carrier": "metabolism / toxins",
        "mechanisms": ["hepatorenal_syndrome", "portal_hypertension_renal", "ammonia_renal_metabolism"],
    },
    "kidney->brain": {
        "carrier": "electrolytes / uremia",
        "mechanisms": ["uremic_encephalopathy", "hyponatremia_cerebral_edema", "acidosis_neuro_depression"],
    },
    "brain->kidney": {
        "carrier": "neurohumoral / blood flow",
        "mechanisms": ["cns_dysautonomia_renal_perfusion", "neurogenic_sodium_imbalance", "stress_hypothalamic_renal"],
    },
    "liver->brain": {
        "carrier": "metabolism / ammonia",
        "mechanisms": ["hepatic_encephalopathy", "ammonia_brain", "portosystemic_ammonia"],
    },
    "brain->liver": {
        "carrier": "neural / circulation",
        "mechanisms": ["cns_hemodynamics_hepatic", "neurogenic_hepatic_ischemia", "autonomic_hepatic_perfusion"],
    },
    "liver->intestine": {
        "carrier": "portal / bile",
        "mechanisms": ["portal_hypertension_gut", "bile_deficiency_malabsorption", "liver_circulation_gut_barrier"],
    },
    "intestine->liver": {
        "carrier": "portal / nutrients",
        "mechanisms": ["portal_blood_nutrient_hepatic", "gut_bacteria_liver_metabolism", "intestinal_absorption_hepatic"],
    },
    "heart->intestine": {
        "carrier": "perfusion / congestion",
        "mechanisms": ["low_perfusion_gut", "venous_congestion_gut", "cardiac_output_splanchnic"],
    },
    "intestine->heart": {
        "carrier": "inflammation / toxins",
        "mechanisms": ["gut_translocation_cardiac", "endotoxin_myocardial", "inflammatory_cytokine_cardiac"],
    },
}


def is_allowed_mechanism(edge, mechanism_code):
    """Check whether mechanism_code is in the allowed list for the directed edge."""
    spec = EDGE_MECHANISMS.get(edge)
    if not spec:
        return False
    return mechanism_code in spec["mechanisms"]


def edge_allowed_mechanisms(edge):
    """Return the list of allowed mechanism codes for a directed edge."""
    spec = EDGE_MECHANISMS.get(edge, {})
    return list(spec.get("mechanisms", []))
