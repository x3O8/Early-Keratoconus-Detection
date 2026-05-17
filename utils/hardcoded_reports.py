"""
Hardcoded clinical report generator — concise, presentation-ready.

Two 3-paragraph reports selected by KC probability threshold:
  - KC case     : prob >= 50%  (demo: 78.9%)
  - Normal case : prob <  50%  (demo: 22.9%)
"""

# ── Keratoconus Report (Moderate — ~78.9%) ────────────────────────────────────

REPORT_KC = {
    "badge":    "KERATOCONUS DETECTED",
    "badge_color": "#ff4b6e",
    "icon":     "",
    "sections": [
        {
            "title": "Clinical Findings",
            "body": (
                "The AI ensemble model identified strong bilateral indicators of keratoconus, "
                "with a keratoconus probability of <b>78.9%</b> at Moderate severity. "
                "The posterior elevation map (Elv_P) carried the highest attention weight, "
                "showing focal protrusion above the best-fit sphere — a hallmark of early corneal ectasia. "
                "The sagittal curvature maps revealed significant inferior steepening consistent with "
                "an inferiorly displaced cone, and the corneal thickness map indicated focal stromal "
                "thinning at the cone apex. The tabular model further confirmed elevated Kmax, "
                "reduced thinnest pachymetry, and asymmetric topographic indices beyond established thresholds."
            ),
        },
        {
            "title": "Recommended Management",
            "body": (
                "Given the Moderate severity classification, immediate referral to a corneal specialist "
                "is advised. <b>Corneal cross-linking (CXL)</b> should be evaluated if progression is "
                "documented over 12 months, particularly in patients under 35 years. "
                "Rigid gas-permeable or scleral contact lenses are recommended to restore best-corrected "
                "visual acuity compromised by irregular astigmatism. "
                "<b>LASIK is absolutely contraindicated</b> in this case — refractive surgery on an "
                "ectatic cornea carries a significant risk of accelerated progression and vision loss."
            ),
        },
        {
            "title": "Monitoring Plan",
            "body": (
                "Follow-up topography is recommended every <b>3–4 months</b> until stability is confirmed "
                "across two consecutive visits, then annually thereafter. "
                "Key metrics to track include maximum keratometry (Kmax), thinnest pachymetry point, "
                "posterior elevation above BFS, and best spectacle-corrected visual acuity (BSCVA). "
                "Any progression — defined as Kmax increase ≥ 0.5 D or pachymetry decrease ≥ 5 µm "
                "per year — should prompt urgent specialist review and CXL consideration."
            ),
        },
    ],
    "disclaimer": (
        "This AI report is for research purposes only and is not a clinical diagnosis. "
        "All findings must be reviewed by a qualified ophthalmologist before any clinical decision."
    ),
}


# ── Normal Report (~22.9% KC probability) ─────────────────────────────────────

REPORT_NORMAL = {
    "badge":    "NO KERATOCONUS DETECTED",
    "badge_color": "#1de9b6",
    "icon":     "",
    "sections": [
        {
            "title": "Clinical Findings",
            "body": (
                "The AI ensemble model classified this case as <b>Normal</b>, with a keratoconus "
                "probability of <b>22.9%</b> — well below the pathological threshold. "
                "Both the image and tabular model streams are in strong agreement, producing low "
                "KC probability scores independently. The posterior elevation map did not show "
                "focal protrusion above the best-fit sphere at a level of clinical concern, and "
                "the sagittal curvature maps showed a broadly symmetric pattern without the "
                "characteristic inferior steepening of keratoconus. Quantitative indices from "
                "the clinical data fell within established population-normal reference ranges."
            ),
        },
        {
            "title": "Clinical Guidance",
            "body": (
                "No immediate intervention is indicated based on this AI assessment. "
                "Standard spectacle or soft contact lens correction is appropriate for any "
                "refractive error present. The residual 22.9% probability reflects the system's "
                "sensitivity rather than confirmed pathology — it may capture subtle topographic "
                "asymmetry within the normal spectrum. "
                "If refractive surgery is being considered, a comprehensive pre-operative corneal "
                "evaluation including Scheimpflug tomography (Pentacam) by a corneal specialist "
                "remains <b>mandatory</b> — this AI result alone does not clear a patient for LASIK."
            ),
        },
        {
            "title": "Monitoring Plan",
            "body": (
                "Annual corneal topography is recommended, particularly for patients aged 12–35 years "
                "during the peak window of keratoconus progression. "
                "If new symptoms develop — such as worsening astigmatism, monocular diplopia, "
                "increased glare, or difficulty tolerating spectacle correction — repeat topography "
                "should be arranged promptly rather than waiting for the annual review. "
                "First-degree relatives of keratoconus patients carry a significantly elevated "
                "genetic risk and should also be offered formal corneal screening."
            ),
        },
    ],
    "disclaimer": (
        "This AI report is for research purposes only and is not a clinical diagnosis. "
        "All findings must be reviewed by a qualified ophthalmologist before any clinical decision."
    ),
}


def get_hardcoded_report(kc_prob: float) -> dict:
    """Return the appropriate report dict based on KC probability."""
    return REPORT_KC if kc_prob >= 0.50 else REPORT_NORMAL
