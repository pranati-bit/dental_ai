"""
The rule that turns OMNI's dentist annotations into our 3 classes.

OMNI annotates every TOOTH with a status code:
  HT  healthy tooth                     TT  tooth torsion (rotated)
  TM  tooth misalignment (crowding,     DO  deep overjet
      spacing)                          MR  mandibular retrusion
  TE  tooth emergence (natural growth)  -- ignored, it is normal development
  IOA, CFOA, OB, FOD  = orthodontic appliances (treatment already started)

Our rule (a design choice, justify it in the report):
  * Any appliance code on the patient  -> EXCLUDED (we predict need BEFORE treatment)
  * many rotated/misaligned FRONT teeth (>= likely_min)  -> braces_likely
  * a few rotated/misaligned FRONT teeth                 -> borderline
  * none                                                 -> no_need
  * optional (use_severe=True): DO or MR on any view also gives braces_likely.
    Off by default, because a frontal photo cannot show overjet or jaw position.
"""
APPLIANCE = {"IOA", "CFOA", "OB", "FOD"}


def label_patient(frontal, patient, borderline_min, likely_min, use_severe=False):
    """frontal: Counter(code -> boxes) of the frontal image.
    patient: Counter over ALL views of the same patient."""
    problem = frontal.get("TT", 0) + frontal.get("TM", 0)
    severe = patient.get("DO", 0) + patient.get("MR", 0)
    out = dict(label=None, excluded="", problem=problem, severe=int(severe > 0))
    if any(patient.get(c, 0) for c in APPLIANCE):
        out["excluded"] = "appliance"
    elif sum(frontal.values()) == 0:
        out["excluded"] = "no_annotation"
    elif (use_severe and severe > 0) or problem >= likely_min:
        out["label"] = "braces_likely"
    elif problem >= borderline_min:
        out["label"] = "borderline"
    else:
        out["label"] = "no_need"
    return out
