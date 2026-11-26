"""
Google Fit
==========

Google-Takeout-Export ("Google Fit"): eine kombinierte Tabelle pro Tag
mit Schritten, Distanz, Herzfrequenz und Schlaf.

Platform info::

    {
        "name": "google_fit",
        "filetypes": ["zip"],
        "languages": ["de", "en"],
        "description": "Google Fit Takeout: daily steps, distance, heart rate and sleep in one table.",
        "time_last_tested": "not yet implemented"
    }
"""
import re
import unicodedata
import zipfile
from collections import Counter
from typing import Callable

import pandas as pd

from port.api.d3i_props import ExtractionResult
from port.helpers.extraction_helpers import ZipArchiveReader
from port.helpers.flow_builder import FlowBuilder
from port.helpers.table_extractor import load_port_config, run_extraction
from port.helpers.validate import StatusCode, ValidateInput

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

FIT_FOLDER = re.compile(r"(^|/)Google Fit/", re.IGNORECASE)
FIT_CSV = re.compile(r"(^|/)Google Fit/[^/]+\.csv$", re.IGNORECASE)   # jede CSV im Fit-Baum
SESSION_JSON = re.compile(r"(^|/)Google Fit/(Alle Sitzungen|All sessions)/[^/]+\.json$", re.IGNORECASE)

# Dateiname der Intraday-CSVs: "2026-10-15.csv" -> Datum aus dem Namen
DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)

# Zeitzonen-Offset aus dem Sitzungs-Dateinamen: "2026-10-14T22_41_00+02_00_SLEEP.json"
TZ_OFFSET = re.compile(r"([+-])(\d{2})_(\d{2})_\d{2}\.json$")

# Zielspalte -> mögliche Spaltennamen im Export (de/en, bestätigt an deinen Dateien).
DAILY_COLUMNS = {
    "date":       ["Datum", "Date"],
    "steps":      ["Schrittzahl", "Step count"],
    "distance_m": ["Distanz (m)", "Distance (m)"],
    "hr_avg_bpm": ["Durchschnittliche Herzfrequenz (bpm)", "Average heart rate (bpm)"],
    "hr_max_bpm": ["Maximale Herzfrequenz (bpm)", "Max heart rate (bpm)"],
    "hr_min_bpm": ["Minimale Herzfrequenz (bpm)", "Min heart rate (bpm)"],
}
# Spalten der Intraday-CSVs (ohne Datum, das kommt aus dem Dateinamen)
INTRADAY_COLUMNS = {
    "steps":      ["Schrittzahl", "Step count"],
    "distance_m": ["Distanz (m)", "Distance (m)"],
    "hr_avg_bpm": ["Durchschnittliche Herzfrequenz (bpm)", "Average heart rate (bpm)"],
    "hr_max_bpm": ["Maximale Herzfrequenz (bpm)", "Max heart rate (bpm)"],
    "hr_min_bpm": ["Minimale Herzfrequenz (bpm)", "Min heart rate (bpm)"],
}

# Bestätigt an deinen Beispielen (BIKING/SLEEP.json):
KEY_ACTIVITY = "fitnessActivity"
KEY_START = "startTime"
KEY_END = "endTime"


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _pick(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([None] * len(df), index=df.index)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_zip_file(archive) -> ValidateInput:
    status_codes = [
        StatusCode(id=0, description="Google Fit export detected"),
        StatusCode(id=1, description="Not a valid zip file"),
        StatusCode(id=2, description="No Google Fit folder in zip"),
    ]
    v = ValidateInput(status_codes, [])
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            v.archive_members = zf.namelist()
    except zipfile.BadZipFile:
        v.set_current_status_code_by_id(1)
        return v
    if any(FIT_FOLDER.search(_nfc(m)) for m in v.archive_members):
        v.set_current_status_code_by_id(0)
    else:
        v.set_current_status_code_by_id(2)
    return v


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die Tageswerte-CSVs
# ---------------------------------------------------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]  # BOM entfernen
    return df


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _aggregate_intraday(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Intraday-CSVs (15-Minuten-Zeilen, Datum im Dateinamen) -> Tageszeilen."""
    rows = []
    for member, raw in frames:
        m = DATE_IN_NAME.search(_nfc(member))
        if not m:
            continue
        date = m.group(1)
        out = pd.DataFrame({t: _pick(raw, cands) for t, cands in INTRADAY_COLUMNS.items()})
        out = _numeric(out, list(INTRADAY_COLUMNS))
        out = out.dropna(how="all", subset=list(INTRADAY_COLUMNS))  # leere 15-Min-Zeilen weg
        if out.empty:
            continue
        rows.append({
            "date": date,
            # Näherung: Mittelwert der Intervall-Ø (Google gewichtet intern anders,
            # exakte Tageswerte liefert nur die Zusammenfassungs-CSV).
            "hr_avg_bpm": out["hr_avg_bpm"].mean(),
            "hr_max_bpm": out["hr_max_bpm"].max(),
            "hr_min_bpm": out["hr_min_bpm"].min(),
            "steps": out["steps"].sum(),
            "distance_m": out["distance_m"].sum(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def _daily_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Zusammenfassungs-CSV (Spalte 'Datum') -> Tageszeilen."""
    out = pd.DataFrame({t: _pick(raw, cands) for t, cands in DAILY_COLUMNS.items()})
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m-%d", errors="coerce")
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = _numeric(out, [c for c in out.columns if c != "date"])
    return out.dropna(how="all", subset=[c for c in out.columns if c != "date"]) \
              .drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Extractoren
# ---------------------------------------------------------------------------

def _daily_activity(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Tageswerte aus allen Fit-CSVs. Erkennt an der Kopfzeile,
    ob es eine Zusammenfassung (Spalte 'Datum') oder eine
    Intraday-Datei (Spalten 'Beginn'/'Ende') ist."""
    summary_frames, intraday_frames = [], []
    for member in reader.archive_members:
        if not FIT_CSV.search(_nfc(member)):
            continue
        result = reader.csv(member)
        if not (result.found and not result.data.empty):
            continue
        raw = _clean_columns(result.data)
        if any(c in raw.columns for c in DAILY_COLUMNS["date"]):
            summary_frames.append(raw)
        elif "Beginn" in raw.columns or "Ende" in raw.columns:
            intraday_frames.append((member, raw))

    daily = pd.DataFrame()
    if summary_frames:
        daily = pd.concat([_daily_summary(r) for r in summary_frames], ignore_index=True)
    if daily.empty and intraday_frames:
        daily = _aggregate_intraday(intraday_frames)
        if not daily.empty:
            errors["DailyFromIntradayApproximation"] += 1
    if daily.empty:
        errors["NoDailyData"] += 1
    return daily


def _sleep(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Schlaf-Sitzungen (JSON) -> eine Zeile pro Tag, mit Zeitzonen-Korrektur
    aus dem Dateinamen."""
    rows = []
    for member in reader.archive_members:
        if not SESSION_JSON.search(_nfc(member)):
            continue
        result = reader.json(member)
        content = result.data if result.found else None
        items = content if isinstance(content, list) else ([content] if content else [])
        for s in items:
            if not isinstance(s, dict):
                continue
            if str(s.get(KEY_ACTIVITY, "")).lower() != "sleep":
                continue
            start = pd.to_datetime(s.get(KEY_START), errors="coerce")  # UTC
            end = pd.to_datetime(s.get(KEY_END), errors="coerce")      # UTC
            if pd.isna(start) or pd.isna(end):
                continue
            # Offset aus dem Dateinamen ("+02_00"), sonst UTC
            offset = pd.Timedelta(0)
            tz = TZ_OFFSET.search(_nfc(member))
            if tz:
                hours, minutes = int(tz.group(2)), int(tz.group(3))
                offset = pd.Timedelta(hours=hours, minutes=minutes)
                if tz.group(1) == "-":
                    offset = -offset
            end_local = end + offset
            start_local = start + offset
            rows.append({
                "date": end_local.strftime("%Y-%m-%d"),  # Aufwachtag = Zuordnungstag
                "sleep_duration_min": (end - start).total_seconds() / 60,
                "bed_hour": start_local.hour,   # nur Stunden, keine exakten Zeiten
                "wake_hour": end_local.hour,
            })
    if not rows:
        errors["NoSleepSessions"] += 1
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("date", as_index=False).agg(
        sleep_duration_min=("sleep_duration_min", "sum"),
        bed_hour=("bed_hour", "min"),
        wake_hour=("wake_hour", "max"),
    )
    return agg.sort_values("date").reset_index(drop=True)

# Metrik-Namen aus der "aggregate"-Liste der Sitzungs-JSONs (bestätigt an BIKING.json)
METRIC_DISTANCE = "com.google.distance.delta"
METRIC_CALORIES = "com.google.calories.expended"
METRIC_HEART_MIN = "com.google.heart_minutes.summary"
METRIC_ACTIVE_MIN = "com.google.active_minutes"


def _aggregate_metrics(session: dict) -> dict:
    """Liest die aggregate-Liste einer Sitzungs-JSON in ein Dict.
    Werte können floatValue (float) oder intValue (int) sein."""
    metrics = {}
    for entry in session.get("aggregate", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("metricName")
        value = entry.get("floatValue", entry.get("intValue"))
        if name and value is not None:
            try:
                metrics[name] = float(value)
            except (TypeError, ValueError):
                pass
    return metrics


def workouts_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Workouts aus "Alle Sitzungen" (JSON) -> eine Zeile pro Training.

    Table documentation::

        {
          "summary": "One row per workout session with type, duration, distance, calories and heart minutes.",
          "source_file": "Takeout/Google Fit/Alle Sitzungen/*.json",
          "columns": {
            "date": "Day of the workout (YYYY-MM-DD).",
            "activity": "Activity type, e.g. biking (English label from Google).",
            "duration_min": "Duration in minutes.",
            "distance_m": "Distance in metres, if recorded.",
            "calories_kcal": "Calories expended, if recorded.",
            "heart_minutes": "Google Fit heart minutes, if recorded."
          }
        }

    Table config::

        {
          "id": "google_fit_workouts",
          "title": {"en": "Workouts", "de": "Workouts"},
          "description": {
            "en": "One row per workout: type, duration, distance, calories and heart minutes.",
            "de": "Eine Zeile pro Training: Typ, Dauer, Distanz, Kalorien und Herzminuten."
          },
          "headers": {
            "date":          {"en": "Date", "de": "Datum"},
            "activity":      {"en": "Activity", "de": "Aktivität"},
            "duration_min":  {"en": "Duration (min)", "de": "Dauer (min)"},
            "distance_m":    {"en": "Distance (m)", "de": "Distanz (m)"},
            "calories_kcal": {"en": "Calories (kcal)", "de": "Kalorien (kcal)"},
            "heart_minutes": {"en": "Heart minutes", "de": "Herzminuten"}
          }
        }
    """
    rows = []
    for member in reader.archive_members:
        if not SESSION_JSON.search(_nfc(member)):
            continue
        result = reader.json(member)
        content = result.data if result.found else None
        items = content if isinstance(content, list) else ([content] if content else [])
        for s in items:
            if not isinstance(s, dict):
                continue
            activity = str(s.get(KEY_ACTIVITY, "")).strip().lower()
            if activity in ("", "sleep"):          # Schlaf gehört zu daily_fit_to_df
                continue
            start = pd.to_datetime(s.get(KEY_START), errors="coerce")
            end = pd.to_datetime(s.get(KEY_END), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            metrics = _aggregate_metrics(s)
            rows.append({
                "date": start.strftime("%Y-%m-%d"),
                "activity": activity,
                "duration_min": round((end - start).total_seconds() / 60, 1),
                "distance_m": metrics.get(METRIC_DISTANCE),
                "calories_kcal": metrics.get(METRIC_CALORIES),
                "heart_minutes": metrics.get(METRIC_HEART_MIN),
            })
    if not rows:
        errors["NoWorkoutSessions"] += 1
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)


def daily_fit_to_df(reader: ZipArchiveReader, errors: Counter) -> pd.DataFrame:
    """Kombinierte Tages-Tabelle: Aktivität + Schlaf, eine Zeile pro Tag.

    Table documentation::

        {
          "summary": "One row per day: steps, distance, heart rate and sleep.",
          "source_file": "Takeout/Google Fit/Tägliche Aktivitätswerte/*.csv and Alle Sitzungen/*.json",
          "columns": {
            "date": "Calendar day (YYYY-MM-DD).",
            "steps": "Step count of the day.",
            "distance_m": "Distance in metres.",
            "hr_avg_bpm": "Average heart rate (bpm).",
            "hr_max_bpm": "Maximum heart rate (bpm).",
            "hr_min_bpm": "Minimum heart rate (bpm).",
            "sleep_duration_min": "Total sleep duration of that night, in minutes.",
            "bed_hour": "Hour of falling asleep, local time (0-23).",
            "wake_hour": "Hour of waking up, local time (0-23)."
          }
        }

    Table config::

        {
          "id": "google_fit_daily",
          "title": {"en": "Daily activity & sleep", "de": "Tagesaktivität & Schlaf"},
          "description": {
            "en": "One row per day: steps, distance, heart rate and sleep. You can remove rows before donating.",
            "de": "Eine Zeile pro Tag: Schritte, Distanz, Herzfrequenz und Schlaf. Sie können Zeilen entfernen, bevor Sie spenden."
          },
          "headers": {
            "date": {"en": "Date", "de": "Datum"},
            "steps": {"en": "Steps", "de": "Schritte"},
            "distance_m": {"en": "Distance (m)", "de": "Distanz (m)"},
            "hr_avg_bpm": {"en": "Heart rate avg (bpm)", "de": "Herzfrequenz Ø (bpm)"},
            "hr_max_bpm": {"en": "Heart rate max (bpm)", "de": "Herzfrequenz max (bpm)"},
            "hr_min_bpm": {"en": "Heart rate min (bpm)", "de": "Herzfrequenz min (bpm)"},
            "sleep_duration_min": {"en": "Sleep (min)", "de": "Schlaf (min)"},
            "bed_hour": {"en": "Bedtime (h)", "de": "Einschlafen (h)"},
            "wake_hour": {"en": "Waking (h)", "de": "Aufwachen (h)"}
          }
        }
    """
    daily = _daily_activity(reader, errors)
    sleep = _sleep(reader, errors)

    if daily.empty and sleep.empty:
        return pd.DataFrame()
    if daily.empty:
        return sleep
    if sleep.empty:
        return daily
    # Outer-Join: Tage ohne Schlaf bzw. ohne Aktivität bleiben mit leeren Feldern
    out = daily.merge(sleep, on="date", how="outer").sort_values("date")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Registry und Verdrahtung
# ---------------------------------------------------------------------------

EXTRACTOR_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "daily_fit_to_df": daily_fit_to_df,
    "workouts_to_df": workouts_to_df,
}


def extraction(archive, validation: ValidateInput) -> ExtractionResult:
    config = load_port_config(EXTRACTOR_REGISTRY, "google_fit")
    errors: Counter = Counter()
    reader = ZipArchiveReader(archive, validation.archive_members, errors)
    return run_extraction(reader, errors, config)


class GoogleFitFlow(FlowBuilder):
    def __init__(self, session_id: str):
        super().__init__(session_id, "google_fit")

    def validate_file(self, file) -> ValidateInput:
        return validate_zip_file(file)

    def extract_data(self, file, validation: ValidateInput) -> ExtractionResult:
        return extraction(file, validation)


def process(session_id: str):
    return GoogleFitFlow(session_id).start_flow()
