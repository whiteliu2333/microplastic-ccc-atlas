# -*- coding: utf-8 -*-
"""
Microplastic CCC Prediction — restrained scientific SaaS interface.

UI only: the original saved stacking model, preprocessing, feature order,
raw categorical option values, validation rules and CCC transformation are
unchanged. All 13 original fields are retained in three panels (4 / 5 / 4).
No values are guessed or silently defaulted.
"""
from __future__ import annotations

import base64
import math
import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import predict_ccc, read_bundle


st.set_page_config(
    page_title="Microplastic CCC Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ccc_stacking.joblib"


@st.cache_resource(show_spinner=False)
def get_bundle(path: str) -> dict:
    return read_bundle(path)


if not MODEL_PATH.is_file():
    st.error("Trained stacking model not found: models/ccc_stacking.joblib")
    st.stop()

try:
    bundle = get_bundle(str(MODEL_PATH))
except Exception as exc:
    st.error("The trained Stacking model could not be loaded.")
    st.exception(exc)
    st.stop()


# The original model feature order and trained categorical mappings are preserved.
COLUMNS = list(bundle["model_columns"])

# The 13 unchanged model inputs, redistributed into a 4 / 5 / 4 grid.
# These Chinese keys and their associated widget configuration are unchanged.
MAIN_GROUPS = [
    (
        "Solution chemistry",
        [
            ("阳离子价态 X1", "Cation type", "Select cation"),
            ("阴离子类型", "Anion type", "Select anion"),
            ("PH X4", "pH", "Enter value"),
            ("温度℃", "Temperature (°C)", "Enter value"),
        ],
    ),
    (
        "Microplastic properties",
        [
            ("塑料种类 X6", "Plastic type", "Select polymer type"),
            ("颗粒形态 X3", "Particle type", "Select particle type"),
            ("塑料表面修饰类型 X5", "Plastic surface modification type", "Select surface modification"),
            ("Zeta电位mV X2", "Initial Zeta potential (mV) (Recommended value: −35 mV)", "Enter value"),
            ("初始粒径nm X8", "Initial particle size (nm) (Recommended value: 100 nm)", "Enter value"),
        ],
    ),
    (
        "Exposure conditions",
        [
            ("塑料浓度mg/L X7", "Plastic concentration (mg L⁻¹)", "Enter value"),
            ("共存物质种类 X9", "Coexisting species type", "Select coexisting species"),
            ("共存物质浓度mg/L X10", "Coexisting species concentration (mg L⁻¹)", "Enter value"),
            ("光照时间h X11", "Irradiation time (h)", "Enter value"),
        ],
    ),
]

FIELD_LABELS = {
    feature: label
    for _, fields in MAIN_GROUPS
    for feature, label, _ in fields
}

if (
    len(FIELD_LABELS) != 13
    or len(COLUMNS) != 13
    or set(FIELD_LABELS) != set(COLUMNS)
):
    st.error("The app's 13 variables do not match this trained model.")
    st.stop()

# Only shown inside numeric input help tooltips, never inside inputs themselves.
NUMERIC_UNITS = {
    "温度℃": "°C",
    "Zeta电位mV X2": "mV",
    "塑料浓度mg/L X7": "mg L⁻¹",
    "初始粒径nm X8": "nm",
    "共存物质浓度mg/L X10": "mg L⁻¹",
    "光照时间h X11": "h",
}

SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# Display aliases do not change the raw values used by the trained model.
ALIASES = {
    "无": "None",
    "MPs": "Microplastics (MPs)",
    "NPs": "Nanoplastics (NPs)",
    "BSA": "Bovine serum albumin (BSA)",
    "CIP": "Ciprofloxacin (CIP)",
    "SMX": "Sulfamethoxazole (SMX)",
    "EPS": "Extracellular polymeric substances (EPS)",
    "FA": "Fulvic acid (FA)",
    "HA": "Humic acid (HA)",
    "NOM": "Natural organic matter (NOM)",
    "SRNOM": "Suwannee River natural organic matter (SRNOM)",
    "CeO2 NPs": "CeO₂ nanoparticles",
    "Fe(OH)3": "Fe(OH)₃",
    "amidine": "Amidine",
    "amino": "Amino",
    "amino groups": "Amino groups",
    "carboxyl": "Carboxyl",
    "ketene": "Ketene",
    "sulfate": "Sulfate",
    "Sulfonyl": "Sulfonyl",
    "bovine trypsin": "Bovine trypsin",
    "nanosized activated carbon": "Nanosized activated carbon",
    "soot nanoparticles": "Soot nanoparticles",
    "kaolinite": "Kaolinite",
    "montmorillonite": "Montmorillonite",
}
HIDE_CATEGORIES = {"未报告", "未报道", "<missing>", "not reported", "not specified"}

# Virtual UI options. Never feed these sentinel strings into the trained model.
OTHER_MONOVALENT = "__ccc_other_monovalent__"
OTHER_DIVALENT = "__ccc_other_divalent__"
OTHER_ANION = "__ccc_other_anion__"
OTHER_POLYMER = "__ccc_other_polymer__"
NONE_NOT_TRAINED = "__ccc_surface_none_not_trained__"

# These are equal-weight alternative experimental scenarios, not arithmetic
# means of the categorical encodings. The model is run once for each scenario,
# and predictions in mM are averaged after inverse transformation.
AVERAGE_OPTIONS = {
    "\u9633\u79bb\u5b50\u4ef7\u6001 X1": {
        OTHER_MONOVALENT: ("Na(1)", "K(1)"),
        OTHER_DIVALENT: ("Ba(2)", "Ca(2)", "Mg(2)"),
    },
    "\u9634\u79bb\u5b50\u7c7b\u578b": {
        OTHER_ANION: None,  # use every supported anion category
    },
    "\u5851\u6599\u79cd\u7c7b X6": {
        OTHER_POLYMER: ("PBAT", "PE", "PET", "PS"),
    },
}

OTHER_LABELS = {
    OTHER_MONOVALENT: "Other (monovalent)",
    OTHER_DIVALENT: "Other (divalent)",
    OTHER_ANION: "Other",
    OTHER_POLYMER: "Other",
    NONE_NOT_TRAINED: "None (unavailable in model)",
}

SURFACE_FEATURE = "\u5851\u6599\u8868\u9762\u4fee\u9970\u7c7b\u578b X5"
SURFACE_UNREPORTED_MARKERS = ("\u672a\u62a5\u544a", "\u672a\u62a5\u9053", "not reported", "not specified", "<missing>")

# Values are based on the 507 rows of Data2.xlsx: zeta median -34 mV,
# most frequent -37.7 mV; particle-size median/mode 100 nm.
RECOMMENDED_NUMERIC = {
    "Zeta\u7535\u4f4dmV X2": "-35 mV",
    "\u521d\u59cb\u7c92\u5f84nm X8": "100 nm",
}

SELECT_HELP = {
    "\u9633\u79bb\u5b50\u4ef7\u6001 X1": (
        "Other (monovalent): equal-weight mean of Na+ and K+ model predictions. "
        "Other (divalent): mean of Ba2+, Ca2+, and Mg2+ predictions."
    ),
    "\u9634\u79bb\u5b50\u7c7b\u578b": (
        "Other: equal-weight mean across all anion categories included in the trained model."
    ),
    "\u5851\u6599\u79cd\u7c7b X6": (
        "Other: equal-weight mean of the PBAT, PE, PET, and PS model predictions."
    ),
    SURFACE_FEATURE: (
        "No confirmed unmodified surface samples are present in the source data. "
        "None uses the model's unreported modification category as an approximation."
    ),
}


def raw_surface_none_option():
    """Prefer a real no-modification code, else use the unreported-data proxy."""
    available = list(bundle["category_options"].get(SURFACE_FEATURE, []))
    for raw in available:
        if str(raw).strip().casefold() in {"\u65e0", "none", "unmodified", "no modification"}:
            return raw, False
    for marker in SURFACE_UNREPORTED_MARKERS:
        for raw in available:
            if str(raw).strip().casefold() == marker.casefold():
                return raw, True
    return NONE_NOT_TRAINED, True


def ionic_parts(raw: str):
    """Extract ion symbol and valence from Na(1) or 1(Na)."""
    text = str(raw).strip()
    match = re.fullmatch(r"([A-Z][a-z]?)\((\d+)\)", text)
    if match:
        return match.group(1), int(match.group(2))
    match = re.fullmatch(r"(\d+)\(([A-Z][a-z]?)\)", text)
    if match:
        return match.group(2), int(match.group(1))
    return text, 999


def mass_fraction(raw: str):
    """Read e.g. <3kDa SRHA, 3-10kDa BCHA, or 10-30k Da HA."""
    match = re.fullmatch(
        r"\s*([<>]?\s*\d+(?:\s*-\s*\d+)?)\s*k\s*Da\s+([A-Za-z]+)\s*",
        str(raw), flags=re.IGNORECASE,
    )
    if match is None:
        return None
    interval = re.sub(r"\s+", "", match.group(1)).replace("-", "~")
    species = match.group(2).upper()
    bound = float(re.search(r"\d+", interval).group())
    if interval.startswith("<"):
        bound = -1
    elif interval.startswith(">"):
        bound += .01
    return species, interval, bound


def scientific_display(feature: str, raw_value: str) -> str:
    """Make clean English/math labels without changing the model's X category."""
    raw = str(raw_value).strip()
    if raw in OTHER_LABELS:
        return OTHER_LABELS[raw]
    if feature == SURFACE_FEATURE and raw == str(raw_surface_none_option()[0]):
        return "None"
    if feature == "阳离子价态 X1":
        element, charge = ionic_parts(raw)
        if charge != 999:
            exponent = "" if charge == 1 else str(charge).translate(SUPERSCRIPT)
            return f"{element}{exponent}⁺"

    if feature == "阴离子类型":
        match = re.fullmatch(r"(.+?)\((\d+)\)", raw)
        if match:
            formula, amount = match.groups()
            charge = int(amount)
            exponent = "" if charge == 1 else str(charge).translate(SUPERSCRIPT)
            return formula.translate(SUBSCRIPT) + exponent + "⁻"

    if raw in ALIASES:
        return ALIASES[raw]

    fraction = mass_fraction(raw)
    if fraction:
        species, interval, _ = fraction
        return f"{species} ({interval} kDa)"

    carboxyl = re.fullmatch(
        r"carboxyl\s+(\d+(?:\.\d+)?)\s*mmol\s*g-1",
        raw, flags=re.IGNORECASE,
    )
    if carboxyl:
        return f"Carboxyl ({carboxyl.group(1)} mmol g⁻¹)"

    dbc = re.fullmatch(r"DBC\((.+)\)", raw)
    if dbc:
        return "DBC (" + dbc.group(1) + ")"

    return raw[:1].upper() + raw[1:]


def scientific_sort(feature: str, raw_value: str):
    """Sort display labels A–Z; put Other near the end and None last."""
    label = scientific_display(feature, raw_value).strip()
    normalized = label.casefold()
    is_none = normalized == "none" or normalized.startswith("none (")
    is_other = normalized == "other" or normalized.startswith("other (")
    group = 2 if is_none else (1 if is_other else 0)
    return (group, normalized, str(raw_value).casefold())


def option_list(feature: str):
    """Display normal categories A–Z, followed by Other, then None."""
    allowed = [
        raw for raw in bundle["category_options"][feature]
        if str(raw).strip().casefold() not in HIDE_CATEGORIES
    ]

    # Add virtual UI choices before sorting; Other and None are placed last
    # without changing the underlying model category codes.
    if feature == SURFACE_FEATURE:
        none_raw, _is_proxy = raw_surface_none_option()
        if none_raw not in allowed:
            allowed.append(none_raw)
    elif feature == "\u9633\u79bb\u5b50\u4ef7\u6001 X1":
        allowed.extend([OTHER_MONOVALENT, OTHER_DIVALENT])
    elif feature == "\u9634\u79bb\u5b50\u7c7b\u578b":
        allowed.append(OTHER_ANION)
    elif feature == "\u5851\u6599\u79cd\u7c7b X6":
        allowed.append(OTHER_POLYMER)

    return sorted(allowed, key=lambda raw: scientific_sort(feature, raw))


def expanded_scenarios(row: dict) -> tuple[list[dict], bool]:
    """Expand each Other option into trained alternatives of equal weight.

    When two or three Other inputs are selected, the Cartesian product is
    evaluated, then the CCC predictions are averaged in mM, not raw encodings.
    """
    scenarios = [dict(row)]
    for feature, virtual_groups in AVERAGE_OPTIONS.items():
        selected = row[feature]
        if selected not in virtual_groups:
            continue
        trained = tuple(bundle["category_options"][feature])
        requested = virtual_groups[selected]
        if requested is None:
            # Existing allowed anion options, including rare ions. Exactly
            # one aggregate Other entry is shown in the UI.
            requested = tuple(
                raw for raw in option_list(feature) if raw != OTHER_ANION
            )
        missing = [value for value in requested if value not in trained]
        if missing or not requested:
            raise ValueError(
                f"The trained model does not support all categories required "
                f"for {FIELD_LABELS[feature]} Other: {', '.join(map(str, missing))}."
            )
        scenarios = [
            dict(scenario, **{feature: alternative})
            for scenario in scenarios
            for alternative in requested
        ]

    # A source row tagged 'not reported' does not prove absence of surface
    # modification. Show that scientific limitation to the user explicitly.
    source, is_proxy = raw_surface_none_option()
    if row[SURFACE_FEATURE] == NONE_NOT_TRAINED:
        raise ValueError(
            "The trained model contains no unmodified or unreported surface "
            "category, so a None prediction is not available."
        )
    surface_proxy = bool(is_proxy and row[SURFACE_FEATURE] == source)
    return scenarios, surface_proxy


def predict_scenario_average(scenarios: list[dict]) -> float:
    """Average full-model CCC predictions of equal-weight scenarios in mM."""
    X = pd.DataFrame(scenarios, columns=COLUMNS)
    if len(scenarios) == 1:
        values, _ = predict_ccc(bundle, X)
        return float(values[0])

    # Standard sklearn-based pipelines support batch prediction. If a custom
    # predictor happens to support only a single row, fall back to single-row
    # calls without changing the averaging method.
    try:
        predictions, _ = predict_ccc(bundle, X)
        scores = pd.Series(predictions).astype(float).to_numpy()
        if len(scores) != len(scenarios):
            raise ValueError("The model returned an unexpected number of results.")
    except Exception:
        scores = []
        for scenario in scenarios:
            one = pd.DataFrame([scenario], columns=COLUMNS)
            values, _ = predict_ccc(bundle, one)
            scores.append(float(values[0]))
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("The model returned a non-finite CCC prediction.")
    return float(sum(float(value) for value in scores) / len(scores))



# ----------------------------------------------------------------------------
# User input state, immediate field warnings and model prediction
# ----------------------------------------------------------------------------
def field_key(feature: str) -> str:
    kind = "cat" if feature in bundle["categorical_columns"] else "num"
    base_key = f"ccc_{kind}_{feature}"
    # Streamlit may restore deleted widget values from the browser on rerun.
    # Changing the widget identity when Reset is pressed guarantees that
    # every selectbox/number_input is rebuilt from its empty default.
    reset_version = st.session_state.get("ccc_reset_version", 0)
    return f"{base_key}__reset_{reset_version}" if reset_version else base_key


def invalidate_prediction():
    """Discard an old prediction as soon as any parameter is changed."""
    st.session_state.pop("ccc_last_result", None)
    st.session_state.pop("ccc_prediction_error", None)


def clear_inputs():
    """Reset all 13 widgets and clear any previous prediction/validation."""
    # The Reset button calls this function BEFORE Streamlit rerenders the page.
    # First remember/remove old widget state, then give all widgets fresh keys.
    old_keys = [field_key(feature) for feature in COLUMNS]
    st.session_state["ccc_reset_version"] = (
        st.session_state.get("ccc_reset_version", 0) + 1
    )
    for key in old_keys:
        st.session_state.pop(key, None)
    st.session_state.pop("ccc_last_result", None)
    st.session_state.pop("ccc_prediction_error", None)
    st.session_state.pop("ccc_attempted", None)
    st.session_state.pop("ccc_show_extra", None)


def numeric_range(feature: str) -> tuple[float, float]:
    stat = bundle["numeric_summary"][feature]
    return float(stat["min"]), float(stat["max"])


def scientific_number(value: float) -> str:
    """Typeset actual negative numbers using the mathematical minus sign."""
    return f"{value:g}".replace("-", "−")


def training_range_label(feature: str) -> str:
    low, high = numeric_range(feature)
    unit = NUMERIC_UNITS.get(feature, "")
    return f"{scientific_number(low)}~{scientific_number(high)}" + (
        f" {unit}" if unit else ""
    )


def out_of_range(row: dict) -> list[str]:
    """Same trained numeric bounds and same warning-only policy as before."""
    outside = []
    for feature in bundle["numeric_columns"]:
        value = row.get(feature)
        if value is None:
            continue
        try:
            number = float(value)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(number):
            continue
        low, high = numeric_range(feature)
        if number < low or number > high:
            outside.append(FIELD_LABELS[feature])
    return outside


def is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def numeric_field_feedback(feature: str, value) -> str | None:
    """Return the single most important message to show beneath this input."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Please enter a valid number."

    if not math.isfinite(number):
        return "Please enter a finite number."

    # Keep the original prediction eligibility rules. Show the errors near
    # the relevant input instead of at the bottom of the page.
    if feature == "初始粒径nm X8" and number <= 0:
        return "Initial particle size must be greater than 0."
    if feature in {
        "塑料浓度mg/L X7",
        "共存物质浓度mg/L X10",
        "光照时间h X11",
    } and number < 0:
        return "This value cannot be negative."
    if (
        feature == "共存物质浓度mg/L X10"
        and st.session_state.get(field_key("共存物质种类 X9")) == "无"
        and number != 0
    ):
        return "Enter 0 when coexisting species is None."

    low, high = numeric_range(feature)
    if number < low or number > high:
        # Not a hard block: the original predictor allowed extrapolation.
        return (
            f"Outside training range ({training_range_label(feature)}). "
            "Prediction may be less reliable."
        )
    return None


def field_input(feature: str, label: str, placeholder: str):
    """Native widgets, same model feature keys/options and numeric constraints."""
    if feature in bundle["categorical_columns"]:
        value = st.selectbox(
            label=label,
            options=option_list(feature),
            index=None,
            format_func=lambda item, f=feature: scientific_display(f, item),
            placeholder=placeholder,
            key=field_key(feature),
            on_change=invalidate_prediction,
        )
    else:
        lower, upper = numeric_range(feature)
        unit = NUMERIC_UNITS.get(feature, "")
        tooltip = f"Range: {scientific_number(lower)}~{scientific_number(upper)}"
        if unit:
            tooltip += f" {unit}"
        # Render only the recommended-value suffix as a separately styled span.
        # Keep FIELD_LABELS and the actual model feature names unchanged.
        display_label = label
        if feature in RECOMMENDED_NUMERIC:
            recommendation = (
                f"(Recommended value: {RECOMMENDED_NUMERIC[feature].replace('-', '−')})"
            )
            display_label = label.replace(recommendation, f"*{recommendation}*")
        value = st.number_input(
            label=display_label,
            value=None,
            placeholder=placeholder,
            min_value=None,
            max_value=None,
            step=0.1 if feature in {"PH X4", "Zeta电位mV X2"} else 1.0,
            format="%.2f",
            help=tooltip,
            key=field_key(feature),
            on_change=invalidate_prediction,
        )

    # A permanent helper slot preserves the original layout height before
    # and after Predict CCC. Validation remains under the associated input;
    # it no longer inserts an extra Streamlit block only upon submission.
    feedback_html = ""
    if st.session_state.get("ccc_attempted") and is_missing(value):
        feedback_html = (
            '<div class="ccc-field-error" role="alert">'
            'This field is required.</div>'
        )
    elif feature in bundle["numeric_columns"]:
        feedback = numeric_field_feedback(feature, value)
        if feedback:
            content = escape(feedback)
            if feedback.startswith("Outside training range"):
                content = content.replace(
                    "Outside training range",
                    '<strong class="ccc-range-prefix">Outside training range</strong>',
                    1,
                )
            feedback_html = (
                '<div class="ccc-field-error" role="alert">'
                + content
                + '</div>'
            )

    if not feedback_html and feature == SURFACE_FEATURE and value is not None:
        source, is_proxy = raw_surface_none_option()
        if value == source and is_proxy:
            feedback_html = (
                '<div class="ccc-field-note">'
                'None is approximated using records with unspecified surface modification.'
                '</div>'
            )

    st.markdown(
        '<div class="ccc-feedback-slot" aria-live="polite">'
        + feedback_html
        + '</div>',
        unsafe_allow_html=True,
    )
    return value


def submit_prediction():
    """Button callback runs before UI rendering, so errors/results are in place."""
    st.session_state["ccc_attempted"] = True
    st.session_state.pop("ccc_last_result", None)
    st.session_state.pop("ccc_prediction_error", None)

    values = {feature: st.session_state.get(field_key(feature)) for feature in COLUMNS}
    missing = [feature for feature in COLUMNS if is_missing(values[feature])]
    if missing:
        st.session_state["ccc_prediction_error"] = (
            "Complete the highlighted fields to predict."
        )
        return

    try:
        row = {}
        for feature in COLUMNS:
            value = values[feature]
            if feature in bundle["numeric_columns"]:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"{FIELD_LABELS[feature]} must be a finite number."
                    )
                row[feature] = numeric
            else:
                # Use the raw training category; the displayed label is cosmetic.
                row[feature] = str(value)

        # Same model-specific validation rules as the original app.py.
        if row["初始粒径nm X8"] <= 0:
            raise ValueError("Initial particle size must be greater than zero.")
        for feature in (
            "塑料浓度mg/L X7",
            "共存物质浓度mg/L X10",
            "光照时间h X11",
        ):
            if row[feature] < 0:
                raise ValueError(f"{FIELD_LABELS[feature]} cannot be negative.")
        if (
            row["共存物质种类 X9"] == "无"
            and row["共存物质浓度mg/L X10"] != 0
        ):
            raise ValueError(
                "Coexisting species concentration must be zero when "
                "coexisting species is None."
            )

        scenarios, surface_proxy = expanded_scenarios(row)
        ccc_value = predict_scenario_average(scenarios)
        st.session_state["ccc_last_result"] = {
            "ccc": ccc_value,
            "out_of_range": out_of_range(row),
            "averaged_scenarios": len(scenarios),
            "surface_proxy": surface_proxy,
        }
    except (ValueError, TypeError, KeyError) as exc:
        st.session_state["ccc_prediction_error"] = str(exc)
    except Exception:
        # Keep technical details in the server log rather than adding a long
        # traceback panel below the prediction page.
        import logging
        logging.exception("CCC prediction failed")
        st.session_state["ccc_prediction_error"] = (
            "Prediction failed. Please check the parameters and try again."
        )


def footer_status_html() -> str:
    """A compact result beside the action buttons, never below the main card."""
    result = st.session_state.get("ccc_last_result")
    if result is not None:
        ccc = result["ccc"]
        rendered = f"{ccc:,.3f}" if 0.001 <= ccc < 1e7 else f"{ccc:.3e}"
        caution = (
            '<span class="ccc-result-caution">'
            '<strong class="ccc-caution-prefix">Caution:</strong> '
            'One or more input values fall outside the training range. '
            'The prediction involves extrapolation and may be less reliable.'
            '</span>'
            if result["out_of_range"]
            else ""
        )
        method = (
            f'<span class="ccc-result-method">Mean of {result["averaged_scenarios"]} predictions</span>'
            if result.get("averaged_scenarios", 1) > 1
            else ""
        )
        surface_note = (
            '<span class="ccc-result-method">Surface proxy</span>'
            if result.get("surface_proxy")
            else ""
        )
        notes_html = caution + method + surface_note
        return (
            '<div class="ccc-action-spacer ccc-footer-result" '
            'role="status" aria-live="polite">'
            '<div class="ccc-result-primary">'
            '<span class="ccc-footer-result-label">Predicted CCC</span>'
            f'<span class="ccc-footer-result-value">{escape(rendered)}'
            '<span class="ccc-footer-result-unit">mM</span></span>'
            '</div>'
            + (f'<div class="ccc-result-meta">{notes_html}</div>'
               if notes_html else '')
            + '</div>'
        )

    message = st.session_state.get("ccc_prediction_error")
    if message:
        return (
            '<div class="ccc-action-spacer ccc-footer-error" '
            'role="alert" aria-live="assertive">'
            + escape(message)
            + '</div>'
        )
    return '<div class="ccc-action-spacer" aria-hidden="true"></div>'


# ----------------------------------------------------------------------------
# Visual design: clean, neutral enterprise form
# ----------------------------------------------------------------------------
# The official theme options in .streamlit/config.toml control global widget
# borders and focus colors. The CSS below supplies local layout refinements.
st.markdown(
    """
<style>
:root {
  --ccc-bg: #F6F8FB;
  --ccc-white: #FFFFFF;
  --ccc-text: #17212F;
  --ccc-label: #283445;
  --ccc-muted: #929CAF;
  --ccc-stroke: #DCE2E9;
  --ccc-hover: #AEBAC9;
  --ccc-focus: #596579;
  --ccc-rule: #E9EDF2;
  --ccc-dark: #25282D;
}

/* Page: one quiet white canvas, no title block or decorative icons. */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--ccc-bg) !important;
  color: var(--ccc-text);
  font-family: Inter, -apple-system, BlinkMacSystemFont,
               "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stHeader"] {
  background: transparent !important;
}
#MainMenu, footer {
  visibility: hidden;
}
.block-container {
  max-width: 1480px !important;
  padding: 18px 22px 48px !important;
}

/* Keyed native container: inputs now rerun immediately on committed edits. */

.st-key-ccc_main_card {
  box-sizing: border-box;
  padding: 32px 32px 24px;
  background: var(--ccc-white) !important;
  border: 1px solid #E2E7ED !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 26px rgba(23, 33, 47, 0.045);
  overflow: visible;

  /* 固定表格位置 */
  position: fixed !important;
  top: 0px !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: calc(100% - 44px) !important;
  max-width: 1436px !important;
  margin: 0 auto !important;
  margin-top: 365px !important;
}





/* Three columns with gentle dividers and a shared visual baseline. */
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) {
  gap: 12px !important;
  align-items: stretch;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"] {
  box-sizing: border-box;
  flex: 1 1 0% !important;
  min-width: 0;
  align-self: stretch;
  position: relative;
  padding: 0 22px;
  background: transparent;
  border: none;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"]:first-child {
  padding-left: 0;
  padding-right: 26px;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"]:last-child {
  padding-left: 26px;
  padding-right: 0;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"]:not(:last-child)::after {
  content: "";
  position: absolute;
  right: -6px;
  top: 0;
  bottom: 0;
  width: 1px;
  background: var(--ccc-rule);
  pointer-events: none;
}
.st-key-ccc_main_card
div[data-testid="stColumn"]:has(.ccc-panel-heading) div[data-testid="stVerticalBlock"] {
  gap: 18px !important;
}
.ccc-panel-heading {
  margin: 0 0 4px;
  padding: 0 0 16px;
  border-bottom: 1px solid #EFF2F5;
  font-size: 18px;
  line-height: 1.4;
  font-weight: 650;
  letter-spacing: -0.025em;
  color: var(--ccc-text);
}

/* Input field labels, always readable but not visually heavy. */
.st-key-ccc_main_card div[data-testid="stSelectbox"] label,
.st-key-ccc_main_card div[data-testid="stNumberInput"] label {
  margin-bottom: 7px !important;
}
.st-key-ccc_main_card div[data-testid="stSelectbox"] label p,
.st-key-ccc_main_card div[data-testid="stNumberInput"] label p {
  color: var(--ccc-label) !important;
  font-size: 15px !important;
  line-height: 1.45 !important;
  font-weight: 550 !important;
}

/* Muted slate-blue recommendation text: understated against monochrome UI. */
.st-key-ccc_field_1_3 div[data-testid="stNumberInput"] label p em,
.st-key-ccc_field_1_4 div[data-testid="stNumberInput"] label p em {
  font-style: normal !important;
  color: #68798D !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}

/* Full visible input borders, including the idle state. */
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"] {
  box-sizing: border-box !important;
  min-height: 46px !important;
  height: 46px !important;
  background-color: #FFFFFF !important;
  border: 1px solid var(--ccc-stroke) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  transition: border-color 140ms ease, box-shadow 140ms ease !important;
}
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"]:hover {
  border-color: var(--ccc-hover) !important;
}
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
  border-color: var(--ccc-focus) !important;
  box-shadow: 0 0 0 2px rgba(89, 101, 121, .10) !important;
}

/* Avoid accidental double outlines within the styled native number input. */
.st-key-ccc_main_card div[data-testid="stSelectbox"] input,
.st-key-ccc_main_card div[data-testid="stNumberInput"] input {
  font-size: 14px !important;
  color: var(--ccc-label) !important;
  caret-color: var(--ccc-focus) !important;
}
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"] input {
  min-height: 42px !important;
  height: 42px !important;
  background: transparent !important;
  box-shadow: none !important;
  padding-inline: 12px !important;
}
.st-key-ccc_main_card div[data-testid="stSelectbox"] input::placeholder,
.st-key-ccc_main_card div[data-testid="stNumberInput"] input::placeholder {
  color: var(--ccc-muted) !important;
  opacity: 1;
}
.st-key-ccc_main_card div[data-testid="stNumberInput"] button {
  color: #8290A1 !important;
  background: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
  border-radius: 5px !important;
}
.st-key-ccc_main_card div[data-testid="stNumberInput"] button:hover {
  color: #334155 !important;
  background: #F1F4F8 !important;
}

/*
 * Dropdown menus are portaled outside st.form by Streamlit/BaseWeb.
 * Style the menu itself, not only the selectbox visible input.
 * Keep BaseWeb's native anchor positioning, keyboard support and flip behavior.
 */
[data-baseweb="popover"]:has([role="listbox"]) {
  z-index: 9999 !important;
}
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] [role="listbox"] {
  box-sizing: border-box;
  background: #FFFFFF !important;
  color: var(--ccc-label) !important;
  max-height: 232px !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  border: 1px solid #E4E9F0 !important;
  border-radius: 8px !important;
  box-shadow: 0 8px 24px rgba(20, 33, 50, .10) !important;
  scrollbar-color: #C5CDD7 transparent;
  scrollbar-width: thin;
}
/* Avoid drawing a second heavy border if menu and listbox are nested. */
[data-baseweb="popover"] [data-baseweb="menu"]:has([role="listbox"]) {
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}
[data-baseweb="popover"] [role="option"] {
  box-sizing: border-box !important;
  display: flex;
  align-items: center;
  min-height: 39px !important;
  padding: 9px 14px !important;
  margin: 0 3px !important;
  border-radius: 5px !important;
  background: #FFFFFF !important;
  color: #283445 !important;
  font-size: 14px !important;
  line-height: 1.45 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"]:focus-visible,
[data-baseweb="popover"] [role="option"][data-highlighted="true"] {
  background: #F3F5F8 !important;
}
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
  background: #EDEFF3 !important;
  color: #17212F !important;
}
[data-baseweb="popover"] [role="listbox"]::-webkit-scrollbar,
[data-baseweb="popover"] [data-baseweb="menu"]::-webkit-scrollbar {
  width: 5px;
}
[data-baseweb="popover"] [role="listbox"]::-webkit-scrollbar-thumb,
[data-baseweb="popover"] [data-baseweb="menu"]::-webkit-scrollbar-thumb {
  background: #C5CDD7;
  border-radius: 4px;
}


/* Chemical option typography: prefer Times New Roman for all script glyphs.
   The raw categorical model values are never changed. */
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"],
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] * {
  font-family: "Times New Roman", Times, "Liberation Serif", serif !important;
}

/* Input content has one type scale. Chemical notation retains New Roman. */
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"],
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] :not(svg):not(path),
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"] input,
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] :not(svg):not(path) {
  font-family: "Times New Roman", Times, "Liberation Serif", serif !important;
  font-size: 14px !important;
}

/* Actions: result sits on the same row as the native buttons. */
.ccc-action-divider {
  height: 1px;
  width: 100%;
  margin: 28px 0 14px;
  background: var(--ccc-rule);
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) {
  align-items: center;
  justify-content: flex-end;
  gap: 12px !important;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) > div[data-testid="stColumn"]:first-child {
  flex: 1 1 auto !important;
  min-width: 0;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) > div[data-testid="stColumn"]:nth-last-child(2) {
  flex: 0 0 144px !important;
  width: 144px;
  min-width: 0;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) > div[data-testid="stColumn"]:last-child {
  flex: 0 0 204px !important;
  width: 204px;
  min-width: 0;
}
/* Center the result on the entire form width (not just the first column).
   The two buttons remain aligned to the right on desktop. */
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-footer-result),
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-footer-error) {
  position: relative;
  min-height: 54px;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-footer-result) > div[data-testid="stColumn"]:first-child,
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-footer-error) > div[data-testid="stColumn"]:first-child {
  position: absolute !important;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  flex: none !important;
  width: max-content !important;
  max-width: calc(100% - 400px);
  min-width: 0 !important;
  z-index: 1;
  display: flex;
  justify-content: center;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-footer-result) > div[data-testid="stColumn"]:first-child .stMarkdown,
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-footer-error) > div[data-testid="stColumn"]:first-child .stMarkdown {
  width: max-content;
  max-width: 100%;
}

.st-key-ccc_main_card div[data-testid="stButton"] > button {
  height: 46px !important;
  min-height: 46px !important;
  border-radius: 8px !important;
  font-size: 14px !important;
  font-weight: 550 !important;
  box-shadow: none !important;
  transition: background-color 140ms ease, border-color 140ms ease !important;
}
.st-key-ccc_reset_action button {
  color: #334155 !important;
  background: #FFFFFF !important;
  border: 1px solid #CFD6E0 !important;
}
.st-key-ccc_reset_action button:hover {
  background: #F7F9FB !important;
  border-color: #ADB9C8 !important;
}
.st-key-ccc_predict_action button {
  color: #FFFFFF !important;
  background: var(--ccc-dark) !important;
  border: 1px solid var(--ccc-dark) !important;
}
.st-key-ccc_predict_action button:hover {
  background: #3B424B !important;
  border-color: #3B424B !important;
}
.st-key-ccc_reset_action button:focus-visible,
.st-key-ccc_predict_action button:focus-visible {
  outline: 2px solid #8794A7 !important;
  outline-offset: 2px !important;
}

/* Only shown when a surface proxy is actually selected. */
.ccc-field-note {
  font-size: 11.5px;
  line-height: 1.4;
  color: #8C6D40;
  padding: 1px 1px 0;
}
.ccc-result-method {
  font-size: 11px;
  color: #678273;
}

/* Compact inline validation: red text under the offending field. */
.ccc-field-error {
  color: #BC302E !important;
  font-size: 11.5px !important;
  line-height: 1.45 !important;
  font-weight: 450;
  margin: 0 1px 1px;
}

/* A slight red border reinforces the message without distracting. */
.st-key-ccc_main_card [class*="st-key-ccc_field_"]:has(.ccc-field-error)
div[data-testid="stNumberInput"] [data-baseweb="input"] {
  border-color: #DDA2A1 !important;
}

/* The input and its helper are a single vertical field unit. */
.st-key-ccc_main_card
div[data-testid="stColumn"]:has(.ccc-panel-heading)
[class*="st-key-ccc_field_"] div[data-testid="stVerticalBlock"] {
  gap: 4px !important;
}
.st-key-ccc_main_card [class*="st-key-ccc_field_"]
div[data-testid="stElementContainer"]:has(.ccc-field-error) {
  margin-top: -2px;
}

/* The result shares the footer row with Reset and Predict CCC. */
/* Muted forest green: success feedback, not a bright AI-style badge. */
.ccc-footer-result {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  min-height: 46px;
  width: fit-content;
  max-width: 100%;
  padding: 6px 16px;
  gap: 6px 12px;
  background: #F4F9F5;
  border: 1px solid #D8E7DD;
  border-radius: 8px;
}
.ccc-footer-result-label {
  color: #547363;
  font-size: 12px;
  font-weight: 550;
  white-space: nowrap;
}
.ccc-footer-result-value {
  color: #236A44;
  font-family: Inter, -apple-system, BlinkMacSystemFont,
               "Segoe UI", Arial, sans-serif;
  font-size: 24px;
  line-height: 1.25;
  font-weight: 700;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.ccc-footer-result-unit {
  color: #48745A;
  font-size: 13px;
  margin-left: 6px;
  font-weight: 500;
}
.ccc-result-caution {
  color: #B42318;
  font-size: 12px;
  line-height: 1.3;
  font-weight: 550;
  max-width: 520px;
  white-space: normal;
}
.ccc-footer-error {
  display: inline-flex;
  align-items: center;
  min-height: 46px;
  max-width: 100%;
  color: #B42318;
  font-size: 12px;
  line-height: 1.4;
  padding: 0 2px;
}
.ccc-action-spacer {
  min-height: 1px;
}
/* Do not create a full-width warning banner or an extra results panel. */
div[data-testid="stAlert"] {
  max-width: 1410px;
  margin-inline: auto;
}

/* When the footer is narrow, keep the result centered above the buttons.
   This rule only changes the footer when a result/error is actually visible. */
@media (max-width: 1250px) {
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-footer-result),
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-footer-error) {
    display: flex;
    flex-wrap: wrap !important;
    gap: 10px !important;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-footer-result) > div[data-testid="stColumn"]:first-child,
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-footer-error) > div[data-testid="stColumn"]:first-child {
    position: static !important;
    transform: none !important;
    flex: 0 0 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    justify-content: center;
  }
}

/* Reduce padding on laptop displays; stack at phone/tablet width. */
@media (max-width: 1180px) {
  .st-key-ccc_main_card {
    padding: 26px 24px 22px;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"] {
    padding-left: 14px;
    padding-right: 14px;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"]:first-child {
    padding-left: 0;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"]:last-child {
    padding-right: 0;
  }
}
@media (max-width: 940px) {
  .block-container {
    padding: 12px 14px 32px !important;
  }
  .st-key-ccc_main_card {
    margin-top: 6px;
    padding: 20px;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) {
    display: flex;
    flex-direction: column;
    gap: 26px !important;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"] {
    width: 100%;
    flex: 1 1 auto !important;
    padding: 0 !important;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) > div[data-testid="stColumn"]:not(:last-child)::after {
    top: auto;
    right: 0;
    left: 0;
    bottom: -15px;
    width: 100%;
    height: 1px;
  }
  .ccc-panel-heading {
    margin-bottom: 0;
    padding-bottom: 12px;
  }
}
@media (max-width: 520px) {
  .block-container {
    padding: 8px 9px 20px !important;
  }
  .st-key-ccc_main_card {
    padding: 16px 15px 18px;
    border-radius: 9px !important;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) {
    display: flex;
    flex-wrap: wrap !important;
    gap: 10px !important;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) > div[data-testid="stColumn"]:first-child {
    display: flex;
    justify-content: center;
    flex: 0 0 100% !important;
    width: 100%;
  }
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) > div[data-testid="stColumn"]:nth-last-child(2),
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) > div[data-testid="stColumn"]:last-child {
    flex: 1 1 0 !important;
    width: auto;
  }
}
@media (max-width: 520px) {
  .st-key-ccc_main_card
  div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer):not(:has(.ccc-footer-result)):not(:has(.ccc-footer-error))
  > div[data-testid="stColumn"]:first-child {
    display: none !important;
  }
}
@media (prefers-reduced-motion: reduce) {
  .st-key-ccc_main_card *,
  .st-key-ccc_main_card *::before,
  .st-key-ccc_main_card *::after {
    transition: none !important;
  }
}

/* Scenic layout: transparent form, with the landscape visible above it. */
.block-container {
  padding-top: clamp(100px, 15vh, 160px) !important;
}

.st-key-ccc_main_card {
  --ccc-rule: rgba(38, 78, 94, 0.18);
  --ccc-muted: #526679;
  background: rgba(255, 255, 255, 0.90) !important;
  border: 0 !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

.st-key-ccc_main_card .ccc-panel-heading {
  border-bottom-color: rgba(38, 78, 94, 0.18) !important;
}

/* Let the water and landscape show through idle input fields. */
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"] {
  background: rgba(255, 255, 255, 0.38) !important;
  border-color: rgba(53, 90, 112, 0.42) !important;
}

.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"]:hover {
  background: rgba(255, 255, 255, 0.53) !important;
  border-color: rgba(53, 90, 112, 0.65) !important;
}

.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
  background: rgba(255, 255, 255, 0.69) !important;
  border-color: var(--ccc-focus) !important;
}

.st-key-ccc_main_card div[data-testid="stSelectbox"] input,
.st-key-ccc_main_card div[data-testid="stNumberInput"] input {
  background: transparent !important;
}

.st-key-ccc_main_card [data-baseweb="select"] [data-baseweb="tag"] {
  background: rgba(255, 255, 255, 0.38) !important;
}

.st-key-ccc_main_card div[data-testid="stNumberInput"] button {
  color: #53697A !important;
}

.st-key-ccc_main_card div[data-testid="stNumberInput"] button:hover {
  background: rgba(255, 255, 255, 0.48) !important;
}

/* Preserve readable, opaque dropdown menus while they are open. */
.st-key-ccc_reset_action button {
  background: rgba(255, 255, 255, 0.40) !important;
  border-color: rgba(53, 90, 112, 0.48) !important;
}
.st-key-ccc_reset_action button:hover {
  background: rgba(255, 255, 255, 0.68) !important;
}

.st-key-ccc_predict_action button {
  background: rgba(37, 40, 45, 0.85) !important;
}
.st-key-ccc_predict_action button:hover {
  background: rgba(37, 40, 45, 0.95) !important;
}

.ccc-footer-result {
  background: rgba(244, 249, 245, 0.86) !important;
}

/* Keep the landscape visible without excessive blank space on small screens. */
@media (max-width: 940px) {
  .block-container {
    padding-top: clamp(76px, 11vh, 100px) !important;
  }
}
@media (max-width: 520px) {
  .block-container {
    padding-top: 76px !important;
  }
}

</style>
""",
    unsafe_allow_html=True,
)



# ----------------------------------------------------------------------------
# FINAL DESKTOP VIEW: fixed-position form, larger readable type, no page scrolling.
#
# The screenshot's visual placement is preserved by the same 350px top margin
# and an approximately 0.88 visual scale (previously auto-computed by JS).
# Larger fonts and tighter vertical spacing keep the footer visible.
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
/* No artificial 185vh page and no scrollbar inside the white prediction card. */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    overflow-x: hidden !important;
}
[data-testid="stMain"] {
    height: 100dvh !important;
    overflow-y: hidden !important;
    overflow-x: hidden !important;
    background-color: #AEE3EC !important;
    background-image:
        linear-gradient(180deg, #B7E8FB 0%, #A1E3EF 48%, #B9EAD5 100%) !important;
    background-size: 100% 100% !important;
    background-position: center top !important;
    background-repeat: no-repeat !important;
}
[data-testid="stMain"] .block-container {
    min-height: 100dvh !important;
}

/* This block retains the user's original fixed position and 350px setting. */
.st-key-ccc_main_card {
    position: fixed !important;
    top: 0px !important;
    left: 50% !important;
    margin-top: 365px !important;
    transform: translateX(-50%) scale(0.88) !important;
    transform-origin: top center !important;
    max-height: none !important;
    overflow: visible !important;
    overflow-x: visible !important;
    overflow-y: visible !important;
    z-index: 20 !important;
    /* Keep the existing white card background, rounding and border styles. */
    padding: 28px 32px 21px !important;
}

/* Panel titles: 18px -> 22px */
.st-key-ccc_main_card .ccc-panel-heading {
    font-size: 22px !important;
    line-height: 1.32 !important;
    padding-bottom: 12px !important;
    margin-bottom: 3px !important;
}

/* Keep the same three-column form without growing the card vertically. */
.st-key-ccc_main_card
div[data-testid="stColumn"]:has(.ccc-panel-heading)
div[data-testid="stVerticalBlock"] {
    gap: 14px !important;
}

/* Field labels: 15px -> 18px */
.st-key-ccc_main_card div[data-testid="stSelectbox"] label,
.st-key-ccc_main_card div[data-testid="stNumberInput"] label {
    margin-bottom: 5px !important;
}
.st-key-ccc_main_card div[data-testid="stSelectbox"] label p,
.st-key-ccc_main_card div[data-testid="stNumberInput"] label p {
    font-size: 18px !important;
    line-height: 1.35 !important;
    font-weight: 550 !important;
}

/* Recommended values: 13px -> 15px */
.st-key-ccc_field_1_3 div[data-testid="stNumberInput"] label p em,
.st-key-ccc_field_1_4 div[data-testid="stNumberInput"] label p em {
    font-size: 15px !important;
}

/* Entered numbers, placeholders and selected options: 14px -> 18px. */
.st-key-ccc_main_card div[data-testid="stSelectbox"] input,
.st-key-ccc_main_card div[data-testid="stNumberInput"] input,
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"],
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] :not(svg):not(path),
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"] input {
    font-size: 18px !important;
}

/* The dropdown list renders in an outside portal, so size it separately. */
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] :not(svg):not(path) {
    font-size: 17px !important;
}

/* Buttons: 14px -> 17px. Their original shapes/colours remain unchanged. */
.st-key-ccc_main_card div[data-testid="stButton"] > button,
.st-key-ccc_main_card div[data-testid="stButton"] > button p {
    font-size: 17px !important;
}
.ccc-action-divider {
    margin: 21px 0 11px !important;
}

/* Validation notes must take up real space beneath the input.
   Absolute-positioned errors caused collisions with the next field. */
.st-key-ccc_main_card [class*="st-key-ccc_field_"] {
    position: relative !important;
}
.st-key-ccc_main_card [class*="st-key-ccc_field_"]
div[data-testid="stElementContainer"]:has(.ccc-field-error) {
    position: relative !important;
    inset: auto !important;
    left: auto !important;
    right: auto !important;
    top: auto !important;
    width: auto !important;
    margin: 3px 0 0 !important;
    z-index: auto !important;
    pointer-events: auto !important;
}
.st-key-ccc_main_card div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) {
    min-height: 54px !important;
    align-items: center !important;
}

/* Keep validation and results legible without excessive vertical spacing. */
.ccc-field-error, .ccc-field-note {
    font-size: 13.5px !important;
    line-height: 1.30 !important;
}
.ccc-footer-error, .ccc-footer-result-label,
.ccc-footer-result-unit, .ccc-result-method {
    font-size: 14px !important;
}


/* Result footer: large CCC value at far left; warning never overlays buttons. */
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer) {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    min-height: 84px !important; /* reserved even before Predict: no layout jump */
    gap: 12px !important;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer)
> div[data-testid="stColumn"]:first-child {
    position: static !important;
    inset: auto !important;
    transform: none !important;
    flex: 1 1 0% !important;
    width: auto !important;
    max-width: none !important;
    min-width: 0 !important;
    z-index: auto !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer)
> div[data-testid="stColumn"]:first-child .stMarkdown,
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer)
> div[data-testid="stColumn"]:first-child
div[data-testid="stMarkdownContainer"] {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    text-align: left !important;
}
.st-key-ccc_main_card .ccc-footer-result {
    box-sizing: border-box !important;
    width: fit-content !important;
    max-width: 100% !important;
    min-height: 78px !important;
    padding: 6px 14px !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    flex-direction: row !important;
    justify-content: flex-start !important;
    align-items: center !important;
    gap: 16px !important;
    text-align: left !important;
}
.st-key-ccc_main_card .ccc-footer-result:has(.ccc-result-meta) {
    width: 100% !important;  /* fill ONLY the left result column */
}
.st-key-ccc_main_card .ccc-result-primary {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    flex: 0 0 auto;
    min-width: 180px;
    gap: 1px;
}
.st-key-ccc_main_card .ccc-footer-result-label {
    font-size: 17px !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
}
.st-key-ccc_main_card .ccc-footer-result-value {
    font-size: 42px !important;
    font-weight: 750 !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
}
.st-key-ccc_main_card .ccc-footer-result-unit {
    font-size: 17px !important;
    font-weight: 550 !important;
    margin-left: 6px !important;
}
.st-key-ccc_main_card .ccc-result-meta {
    flex: 1 1 0%;
    min-width: 0;
    max-width: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    gap: 3px;
}
.st-key-ccc_main_card .ccc-result-caution {
    display: block !important;
    color: #B42318 !important;
    font-size: 14px !important;
    font-weight: 550 !important;
    line-height: 1.32 !important;
    max-width: 100% !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
}
.st-key-ccc_main_card .ccc-result-method {
    font-size: 13px !important;
    line-height: 1.25 !important;
}
.st-key-ccc_main_card .ccc-footer-error {
    font-size: 15px !important;
    text-align: left !important;
    max-width: 100% !important;
}

/* The action buttons keep their original widths and their own columns. */
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer)
> div[data-testid="stColumn"]:nth-last-child(2) {
    position: relative !important;
    flex: 0 0 144px !important;
    width: 144px !important;
}
.st-key-ccc_main_card
div[data-testid="stHorizontalBlock"]:has(.ccc-action-spacer)
> div[data-testid="stColumn"]:last-child {
    position: relative !important;
    flex: 0 0 204px !important;
    width: 204px !important;
}

/* On narrower displays, wrap warning *within its column*, not over buttons. */
@media (max-width: 1050px) {
    .st-key-ccc_main_card .ccc-footer-result {
        gap: 10px !important;
        padding-inline: 10px !important;
    }
    .st-key-ccc_main_card .ccc-result-primary {
        min-width: 145px;
    }
    .st-key-ccc_main_card .ccc-footer-result-value {
        font-size: 36px !important;
    }
    .st-key-ccc_main_card .ccc-result-caution {
        font-size: 12.5px !important;
    }
}

/* Keep the original emphasized Caution text in the result footer. */
.st-key-ccc_main_card .ccc-result-caution .ccc-caution-prefix {
    font-size: 20px !important;
    line-height: 1 !important;
    font-weight: 700 !important;
    color: inherit !important;
}

/* ===============================================================
   CCC INPUT PANEL — unified type scale and stable, aligned layout.
   Changes are deliberately limited to the THREE input panels.
   The Predict CCC footer, model, image and card offset are untouched.
   =============================================================== */
.st-key-ccc_main_card {
    --ccc-panel-font: 18px;
    --ccc-note-font: 15px;
}

/* All ordinary visible input-panel text uses precisely 18px:
   column titles, field names, selected values, placeholders, and
   the parenthesized Recommended value notes. */
.st-key-ccc_main_card .ccc-panel-heading,
.st-key-ccc_main_card div[data-testid="stSelectbox"] label p,
.st-key-ccc_main_card div[data-testid="stNumberInput"] label p,
.st-key-ccc_main_card div[data-testid="stNumberInput"] label p em,
.st-key-ccc_main_card div[data-testid="stSelectbox"] input,
.st-key-ccc_main_card div[data-testid="stNumberInput"] input,
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"],
.st-key-ccc_main_card div[data-testid="stSelectbox"] [data-baseweb="select"] :not(svg):not(path),
.st-key-ccc_main_card div[data-testid="stNumberInput"] [data-baseweb="input"] input {
    font-size: var(--ccc-panel-font) !important;
    line-height: 1.30 !important;
}
.st-key-ccc_main_card .ccc-panel-heading {
    font-weight: 650 !important;
}
.st-key-ccc_main_card div[data-testid="stNumberInput"] label p em {
    font-style: normal !important;
    color: #68798D !important;
    font-weight: 500 !important;
    white-space: normal !important;
}

/* Open dropdown items appear in a portal outside the card. */
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] :not(svg):not(path) {
    font-size: 18px !important;
}

/* The complete range warning has ONE slightly smaller type size,
   including its "Outside training range" prefix and entire sentence.
   Required-field feedback follows the same size. */
.st-key-ccc_main_card .ccc-field-error,
.st-key-ccc_main_card .ccc-field-error:has(.ccc-range-prefix),
.st-key-ccc_main_card .ccc-field-error .ccc-range-prefix,
.st-key-ccc_main_card .ccc-field-note {
    font-size: var(--ccc-note-font) !important;
    font-weight: 500 !important;
    line-height: 1.30 !important;
    letter-spacing: normal !important;
    white-space: normal !important;
    overflow-wrap: normal !important;
}
.st-key-ccc_main_card .ccc-field-error .ccc-range-prefix {
    font-weight: 500 !important;
}
/* Keep short warning text directly under its corresponding input.
   It is in normal document flow and pushes later fields down
   if it wraps onto two lines. It can never overlay the next label. */
.st-key-ccc_main_card [class*="st-key-ccc_field_"]
div[data-testid="stVerticalBlock"] {
    gap: 3px !important;
}
.st-key-ccc_main_card [class*="st-key-ccc_field_"]
div[data-testid="stElementContainer"]:has(.ccc-field-error) {
    position: static !important;
    margin-top: 3px !important;
}
.st-key-ccc_main_card .ccc-field-error {
    margin: 0 !important;
    padding: 0 !important;
}

/* Desktop: all three panels stretch to the SAME height.
   The 5-field middle column determines that height. In each 4-field
   side column, space-between distributes the free room into 3 even
   gaps, so the first and LAST input align with the middle column. */
@media (min-width: 941px) {
    .st-key-ccc_main_card
    div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading) {
        align-items: stretch !important;
    }
    .st-key-ccc_main_card
    div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading)
    > div[data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
        align-self: stretch !important;
        min-width: 0 !important;
    }
    .st-key-ccc_main_card
    div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading)
    > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"],
    .st-key-ccc_main_card
    div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading)
    > div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        align-self: stretch !important;
        min-height: 0 !important;
        height: 100% !important;
    }
    .st-key-ccc_main_card
    div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading)
    > div[data-testid="stColumn"]:is(:first-child,:last-child)
    div[data-testid="stVerticalBlock"]:has(.ccc-panel-heading) {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        height: 100% !important;
        justify-content: flex-start !important;
        align-items: stretch !important;
        gap: 14px !important;
    }
    .st-key-ccc_main_card
    div[data-testid="stHorizontalBlock"]:has(.ccc-panel-heading)
    > div[data-testid="stColumn"]:nth-child(2)
    div[data-testid="stVerticalBlock"]:has(.ccc-panel-heading) {
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        gap: 14px !important;
    }

    /* The first field stays aligned with the middle column.
       Auto margins are shared ONLY by subsequent fields:
       side 4th inputs and middle 5th input finish at one baseline. */
    .st-key-ccc_main_card .st-key-ccc_field_0_1,
    .st-key-ccc_main_card .st-key-ccc_field_0_2,
    .st-key-ccc_main_card .st-key-ccc_field_0_3,
    .st-key-ccc_main_card .st-key-ccc_field_1_1,
    .st-key-ccc_main_card .st-key-ccc_field_1_2,
    .st-key-ccc_main_card .st-key-ccc_field_1_3,
    .st-key-ccc_main_card .st-key-ccc_field_1_4,
    .st-key-ccc_main_card .st-key-ccc_field_2_1,
    .st-key-ccc_main_card .st-key-ccc_field_2_2,
    .st-key-ccc_main_card .st-key-ccc_field_2_3 {
        margin-top: auto !important;
    }
}

/* Small screens use natural vertical flow instead of alignment pressure. */
@media (max-width: 940px) {
    .st-key-ccc_main_card
    div[data-testid="stColumn"]:has(.ccc-panel-heading)
    div[data-testid="stVerticalBlock"]:has(.ccc-panel-heading) {
        justify-content: flex-start !important;
        height: auto !important;
    }
}

/* ===============================================================
   FINAL FIX — no hidden Predict CCC result after validation.
   1) Each input ALWAYS reserves the same feedback space.
   2) The outer Streamlit page (not the card) can be scrolled.
   3) Background stays fixed while the page is scrolled.
   4) Original 365px initial offset, font sizes and footer stay.
   =============================================================== */

/* The field's Markdown slot is present even before Predict is clicked. */
.st-key-ccc_main_card .ccc-feedback-slot {
    box-sizing: border-box !important;
    display: block !important;
    min-height: 43px !important;
    padding: 2px 0 0 !important;
    overflow: visible !important;
}
.st-key-ccc_main_card .ccc-feedback-slot .ccc-field-error,
.st-key-ccc_main_card .ccc-feedback-slot .ccc-field-note {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 15px !important;
    line-height: 1.30 !important;
    font-weight: 500 !important;
    white-space: normal !important;
}
.st-key-ccc_main_card .ccc-feedback-slot .ccc-range-prefix {
    font-size: inherit !important;
    line-height: inherit !important;
    font-weight: inherit !important;
}

/* Never position the helper text over its following input. */
.st-key-ccc_main_card [class*="st-key-ccc_field_"]
div[data-testid="stElementContainer"]:has(.ccc-feedback-slot) {
    position: static !important;
    inset: auto !important;
    width: auto !important;
    margin: 0 !important;
}
.st-key-ccc_main_card [class*="st-key-ccc_field_"]
div[data-testid="stVerticalBlock"] {
    gap: 3px !important;
}

/* Allow scrolling on the OUTER right-hand side of the page.
   Previously the app locked the scroll container with overflow:hidden. */
[data-testid="stMain"] {
    height: 100dvh !important;
    min-height: 100dvh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior-y: auto !important;
    scrollbar-width: auto !important;
    scroll-behavior: auto !important;
}
[data-testid="stMain"] .block-container {
    padding-top: 0 !important;
    padding-bottom: 40px !important;
    min-height: 100dvh !important;
}

/* A fixed-position card cannot reveal its clipped footer through outer
   page scrolling. Keep the SAME initial 365px offset but return the card
   to document flow so scrolling reveals the prediction/results area.
   Neither the white card nor its fields gets an internal scrollbar. */
.st-key-ccc_main_card {
    position: relative !important;
    top: auto !important;
    left: auto !important;
    right: auto !important;
    transform: scale(0.88) !important;
    transform-origin: top center !important;
    width: calc(100% - 44px) !important;
    max-width: 1436px !important;
    margin: 365px auto 45px !important;
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    overflow-y: visible !important;
    overflow-x: visible !important;
}

/* Retain the SAME original responsive scales. Their values do not change
   when warning messages appear. */
@media (max-height: 880px) {
    .st-key-ccc_main_card { transform: scale(0.75) !important; }
}
@media (max-height: 740px) {
    .st-key-ccc_main_card { transform: scale(0.56) !important; }
}
@media (max-height: 620px) {
    .st-key-ccc_main_card { transform: scale(0.38) !important; }
}
@media (max-width: 940px) {
    .st-key-ccc_main_card {
        width: calc(100% - 28px) !important;
        margin-top: 120px !important;
    }
    .st-key-ccc_main_card .ccc-feedback-slot {
        min-height: 52px !important;
    }
}

/* Add a little room after the card for the footer and pointer access. */
[data-testid="stMain"] .block-container::after {
    content: "";
    display: block;
    height: 38px;
}

/* Do not rescale this card when Predict CCC adds field warnings. */


/* Height-dependent scaling is defined above in FINAL FIX. */

</style>
""",
    unsafe_allow_html=True,
)

# Keep back.png beside app.py as before. Preserve the screenshot's image zoom.
BACKGROUND_PATH = ROOT / "back.png"
if BACKGROUND_PATH.is_file():
    background_base64 = base64.b64encode(
        BACKGROUND_PATH.read_bytes()
    ).decode("ascii")
    st.markdown(
        f"""
<style>
[data-testid="stMain"] {{
    background-image:
        url("data:image/png;base64,{background_base64}"),
        linear-gradient(180deg, #B7E8FB 0%, #A1E3EF 48%, #B9EAD5 100%) !important;
    background-size: 112% auto, 100% 100% !important;
    background-position: center top, center top !important;
    background-repeat: no-repeat, no-repeat !important;
    background-attachment: fixed, fixed !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Parameters and footer: three columns, 13 inputs, instant per-field feedback.
# ----------------------------------------------------------------------------
form_values = {}
with st.container(key="ccc_main_card", border=False):
    left, middle, right = st.columns(3, gap="medium")
    for group_index, (column, (heading, fields)) in enumerate(
        zip((left, middle, right), MAIN_GROUPS)
    ):
        with column:
            st.markdown(
                f'<div class="ccc-panel-heading">{escape(heading)}</div>',
                unsafe_allow_html=True,
            )
            for field_index, (feature, label, placeholder) in enumerate(fields):
                with st.container(key=f"ccc_field_{group_index}_{field_index}"):
                    form_values[feature] = field_input(feature, label, placeholder)

    st.markdown(
        '<div class="ccc-action-divider"></div>',
        unsafe_allow_html=True,
    )
    status_column, reset_column, predict_column = st.columns(
        [7.5, 1.4, 2.0], gap="small", vertical_alignment="center"
    )
    with status_column:
        st.markdown(footer_status_html(), unsafe_allow_html=True)
    with reset_column:
        st.button(
            "Reset",
            type="secondary",
            use_container_width=True,
            key="ccc_reset_action",
            on_click=clear_inputs,
        )
    with predict_column:
        st.button(
            "Predict CCC",
            type="primary",
            use_container_width=True,
            key="ccc_predict_action",
            on_click=submit_prediction,
        )
