# Databricks notebook source
# MAGIC %md
# MAGIC # Round-trip flight grid (carry-on–aware pricing)
# MAGIC
# MAGIC Busca combinaciones **(fecha salida de ida desde origen, fecha regreso desde destino)** con
# MAGIC [`flights` / `fli`](https://github.com/punitarani/fli). **Modos** (`SEARCH_MODE`): **`VACATION_WINDOW`**
# MAGIC (ventana de vacaciones + días de viaje) y **`DEST_RETURN_DATES`** (rangos de ida y regreso desde destino,
# MAGIC cartesiano, sin widget de duración).
# MAGIC
# MAGIC **`SORT_BY` y aerolíneas:** con **`CHEAPEST`** suele predominar lo más barato (p. ej. Frontier en rutas US).
# MAGIC El default del widget es **`BEST`**. Para excluir aerolíneas, filtrá en Spark (`outbound_airlines_flights`, etc.).
# MAGIC
# MAGIC **429 y coste:** ~**1 + `RT_RECURSION_TOP_N`** POSTs por par. Pacing, retries, `SHOW_ALL`, `MAX_SEARCH_PAIRS` y
# MAGIC `TOP_RESULTS_PER_SEARCH` son **constantes** al inicio del notebook (no widgets).
# MAGIC
# MAGIC **Llegada a destino:** la API usa fecha de **salida del segmento**, no “llegada a SFO el 14”; refiná con
# MAGIC `outbound_arr_date` en Spark si hace falta.
# MAGIC
# MAGIC **Carry-on / escalas:** `BagsFilter(carry_on=True)`; **`TWO_OR_FEWER_STOPS`** fijo en código.
# MAGIC
# MAGIC **Compliance:** API no oficial; respetá términos de Google. Not legal advice.
# MAGIC
# MAGIC **Install:** `flights` 0.x; Python ≥ 3.10. Tras `%pip`, **reiniciá Python** (`typing_extensions` / `Sentinel`).

# COMMAND ----------
# MAGIC %pip install -q 'flights>=0.8.4,<0.9' 'typing_extensions>=4.15.0,<5'

# COMMAND ----------
# MAGIC %md
# MAGIC **Restart Python** after the cell above so the upgraded `typing_extensions` wins on `sys.path` (avoids
# MAGIC `ImportError: cannot import name 'Sentinel' from 'typing_extensions'`). Then **Run all** from the top once.

# COMMAND ----------
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------
from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from urllib.error import HTTPError

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from fli.models import (
    Airport,
    BagsFilter,
    FlightResult,
    FlightSearchFilters,
    FlightSegment,
    MaxStops,
    PassengerInfo,
    SeatType,
    SortBy,
    TripType,
)
from fli.search import SearchFlights

# COMMAND ----------
# --- Tunables (no widgets): pacing, caps, fli RT recursion ---
# Subir SHOW_ALL / RT_RECURSION_TOP_N o bajar delays aumenta 429. Ajustar aquí si hace falta.

MAX_SEARCH_PAIRS = 200
TOP_RESULTS_PER_SEARCH = 35
RT_RECURSION_TOP_N = 6
SHOW_ALL_API_RESULTS = False

MIN_DELAY_SEC = 8.0
MAX_DELAY_SEC = 22.0
MAX_RETRIES = 8
RATE_LIMIT_COOLDOWN_SEC = 90.0

PAIR_HOME_ARRIVAL_SLACK_DAYS = 2
MAX_STOPS = MaxStops.TWO_OR_FEWER_STOPS

# COMMAND ----------
# --- Databricks widgets (YYYYMMDD for dates; dropdowns for enums) ---

def _w_create_text(name: str, default: str) -> None:
    try:
        dbutils.widgets.text(name, default, name)  # noqa: F821
    except Exception:
        pass


def _w_create_dropdown(name: str, default: str, choices: list[str]) -> None:
    try:
        dbutils.widgets.dropdown(name, default, choices, name)  # noqa: F821
    except Exception:
        pass


def _ensure_widgets() -> None:
    _w_create_dropdown(
        "SEARCH_MODE",
        "VACATION_WINDOW",
        ["VACATION_WINDOW", "DEST_RETURN_DATES"],
    )
    _w_create_text("ORIGIN_AIRPORT", "TPA")
    _w_create_text("DESTINATION_AIRPORT", "SFO")
    # VACATION_WINDOW: primer día que podés salir; último día que debés estar de vuelta en casa; duración del viaje.
    _w_create_text("VACATION_FIRST_DAY", "20260601")
    _w_create_text("VACATION_LAST_HOME_DAY", "20260815")
    _w_create_dropdown("MIN_TRIP_DAYS", "10", [str(i) for i in range(1, 22)])
    _w_create_dropdown("MAX_TRIP_DAYS", "14", [str(i) for i in range(1, 31)])
    # DEST_RETURN_DATES: salida desde origen × regreso desde destino (sin widget de “días de duración”).
    _w_create_text("OUTBOUND_DEPART_START", "20260613")
    _w_create_text("OUTBOUND_DEPART_END", "20260614")
    _w_create_text("RETURN_DEPART_DEST_START", "20260622")
    _w_create_text("RETURN_DEPART_DEST_END", "20260624")
    _w_create_dropdown("ADULTS", "1", [str(i) for i in range(1, 7)])
    _w_create_dropdown(
        "SEAT_TYPE",
        "ECONOMY",
        ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"],
    )
    _w_create_dropdown(
        "SORT_BY",
        "BEST",
        [
            "TOP_FLIGHTS",
            "BEST",
            "CHEAPEST",
            "DEPARTURE_TIME",
            "ARRIVAL_TIME",
            "DURATION",
            "EMISSIONS",
        ],
    )
    _w_create_dropdown("EXCLUDE_BASIC_ECONOMY", "true", ["true", "false"])
    _w_create_dropdown("APPLY_TRIP_WINDOW_FILTERS", "false", ["false", "true"])
    _obsolete = (
        "PRICE_CUTOFF_PERCENT",
        "MAX_API_CALLS",
        "MIN_DELAY_SEC",
        "MAX_DELAY_SEC",
        "MAX_RETRIES",
        "RATE_LIMIT_COOLDOWN_SEC",
        "TOP_RESULTS_PER_SEARCH",
        "SHOW_ALL_API_RESULTS",
        "RT_RECURSION_TOP_N",
        "MAX_STOPS",
        "MIN_TRIP_LENGTH_DAYS",
        "MAX_TRIP_LENGTH_DAYS",
        "RETURN_ARRIVAL_HOME_START",
        "RETURN_ARRIVAL_HOME_END",
    )
    for _name in _obsolete:
        try:
            dbutils.widgets.remove(_name)  # noqa: F821
        except Exception:
            pass


_ensure_widgets()

# COMMAND ----------
# --- Read widgets → typed config ---


def _w_get(name: str) -> str:
    return dbutils.widgets.get(name).strip()  # noqa: F821


def parse_ymd(name: str) -> date:
    raw = _w_get(name).replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"{name} must be YYYYMMDD (8 digits), got {_w_get(name)!r}")
    return datetime.strptime(raw, "%Y%m%d").date()


def _w_int(name: str) -> int:
    return int(_w_get(name))


def parse_airport_iata(widget_name: str) -> Airport:
    """Resolve a 3-letter IATA code to ``fli.models.Airport`` (bundled enum)."""
    raw = _w_get(widget_name).strip().upper().replace(" ", "")
    if len(raw) != 3 or not raw.isalpha():
        raise ValueError(
            f"{widget_name} must be exactly 3 letters (IATA), got {_w_get(widget_name)!r}"
        )
    if not hasattr(Airport, raw):
        raise ValueError(
            f"{widget_name}={raw!r} is not in `fli.models.Airport` (unsupported or unknown code in this library)."
        )
    return getattr(Airport, raw)


AIRPORT_ORIGIN = parse_airport_iata("ORIGIN_AIRPORT")
AIRPORT_DEST = parse_airport_iata("DESTINATION_AIRPORT")

SEARCH_MODE = _w_get("SEARCH_MODE").strip().upper().replace(" ", "_")
if SEARCH_MODE not in ("VACATION_WINDOW", "DEST_RETURN_DATES"):
    raise ValueError(
        "SEARCH_MODE must be VACATION_WINDOW or DEST_RETURN_DATES, "
        f"got {_w_get('SEARCH_MODE')!r}"
    )

if SEARCH_MODE == "VACATION_WINDOW":
    VACATION_FIRST_DAY = parse_ymd("VACATION_FIRST_DAY")
    VACATION_LAST_HOME_DAY = parse_ymd("VACATION_LAST_HOME_DAY")
    MIN_TRIP_LENGTH_DAYS = _w_int("MIN_TRIP_DAYS")
    MAX_TRIP_LENGTH_DAYS = _w_int("MAX_TRIP_DAYS")
    if VACATION_LAST_HOME_DAY < VACATION_FIRST_DAY:
        raise ValueError("VACATION_LAST_HOME_DAY must be on or after VACATION_FIRST_DAY")
    if MIN_TRIP_LENGTH_DAYS < 1 or MAX_TRIP_LENGTH_DAYS < MIN_TRIP_LENGTH_DAYS:
        raise ValueError("MIN_TRIP_DAYS / MAX_TRIP_DAYS are invalid")
    last_outbound = VACATION_LAST_HOME_DAY - timedelta(days=MIN_TRIP_LENGTH_DAYS)
    if last_outbound < VACATION_FIRST_DAY:
        raise ValueError(
            "No hay rango de salidas posible: ampliá VACATION_LAST_HOME_DAY o bajá MIN_TRIP_DAYS."
        )
    OUTBOUND_DEPART_START = VACATION_FIRST_DAY
    OUTBOUND_DEPART_END = last_outbound
    RETURN_ARRIVAL_HOME_START = VACATION_FIRST_DAY
    RETURN_ARRIVAL_HOME_END = VACATION_LAST_HOME_DAY
else:
    OUTBOUND_DEPART_START = parse_ymd("OUTBOUND_DEPART_START")
    OUTBOUND_DEPART_END = parse_ymd("OUTBOUND_DEPART_END")
    RETURN_DEPART_DEST_START = parse_ymd("RETURN_DEPART_DEST_START")
    RETURN_DEPART_DEST_END = parse_ymd("RETURN_DEPART_DEST_END")
    if OUTBOUND_DEPART_END < OUTBOUND_DEPART_START:
        raise ValueError("OUTBOUND_DEPART_END must be on or after OUTBOUND_DEPART_START")
    if RETURN_DEPART_DEST_END < RETURN_DEPART_DEST_START:
        raise ValueError("RETURN_DEPART_DEST_END must be on or after RETURN_DEPART_DEST_START")
    # Post-filtro in-notebook: ventana amplia; filtrá por fechas reales en Spark si hace falta.
    RETURN_ARRIVAL_HOME_START = date(1970, 1, 1)
    RETURN_ARRIVAL_HOME_END = date(2100, 12, 31)
    MIN_TRIP_LENGTH_DAYS = 1
    MAX_TRIP_LENGTH_DAYS = 366

ADULTS = _w_int("ADULTS")
SEAT_TYPE = getattr(SeatType, _w_get("SEAT_TYPE"))
SORT_BY = getattr(SortBy, _w_get("SORT_BY"))
EXCLUDE_BASIC_ECONOMY = _w_get("EXCLUDE_BASIC_ECONOMY").lower() in ("1", "true", "yes")
APPLY_TRIP_WINDOW_FILTERS = _w_get("APPLY_TRIP_WINDOW_FILTERS").lower() in ("1", "true", "yes")

# COMMAND ----------
# --- Validation (fail fast) ---

if AIRPORT_ORIGIN == AIRPORT_DEST:
    raise ValueError("ORIGIN_AIRPORT and DESTINATION_AIRPORT must be different")

if SEARCH_MODE == "VACATION_WINDOW" and RETURN_ARRIVAL_HOME_END < RETURN_ARRIVAL_HOME_START:
    raise ValueError("RETURN_ARRIVAL_HOME_END must be on or after RETURN_ARRIVAL_HOME_START")
if MAX_SEARCH_PAIRS < 1:
    raise ValueError("MAX_SEARCH_PAIRS (constant) must be >= 1")
if MIN_DELAY_SEC <= 0 or MAX_DELAY_SEC <= 0 or MAX_DELAY_SEC < MIN_DELAY_SEC:
    raise ValueError("MIN_DELAY_SEC / MAX_DELAY_SEC must be positive and MIN <= MAX")
if MAX_RETRIES < 1:
    raise ValueError("MAX_RETRIES must be >= 1")
if RATE_LIMIT_COOLDOWN_SEC < 0:
    raise ValueError("RATE_LIMIT_COOLDOWN_SEC must be >= 0")
if TOP_RESULTS_PER_SEARCH < 1 or TOP_RESULTS_PER_SEARCH > 200:
    raise ValueError("TOP_RESULTS_PER_SEARCH (constant) must be between 1 and 200")
if RT_RECURSION_TOP_N < 1 or RT_RECURSION_TOP_N > 30:
    raise ValueError("RT_RECURSION_TOP_N must be between 1 and 30 (each step adds Google POSTs)")

today = date.today()
if OUTBOUND_DEPART_START < today:
    raise ValueError("Outbound dates cannot start in the past (fli validates this too).")

# COMMAND ----------
# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("tpa_lax_search")

if MAX_SEARCH_PAIRS > 500:
    log.warning("MAX_SEARCH_PAIRS=%d is high; raises 429 risk and runtime.", MAX_SEARCH_PAIRS)

if SHOW_ALL_API_RESULTS and TOP_RESULTS_PER_SEARCH > 50:
    log.warning(
        "SHOW_ALL_API_RESULTS=true with TOP_RESULTS_PER_SEARCH=%d may slow searches and increase 429s.",
        TOP_RESULTS_PER_SEARCH,
    )

# COMMAND ----------
# --- Candidate (outbound_date, return_depart_date) pairs ---

# En DEST_RETURN_DATES, descarta pares con demasiados días entre salida de ida y regreso desde destino.
MAX_OD_RD_SPAN_DAYS = 90


def daterange(d0: date, d1: date) -> Iterable[date]:
    step = timedelta(days=1)
    cur = d0
    while cur <= d1:
        yield cur
        cur += step


def pair_plausible_for_home_and_trip_length(
    od: date,
    rd: date,
    ret_home_start: date,
    ret_home_end: date,
    min_trip_days: int,
    max_trip_days: int,
    home_slack_days: int,
) -> bool:
    """True si existe algún día de llegada a casa coherente con `rd`, la ventana de widgets y MIN/MAX días de viaje.

    Alineado con el post-filtro: ``trip_length_days ≈ (llegada_casa - día_salida_ida)`` en días de calendario;
    aquí asumimos salida de ida el calendario ``od`` y llegada a casa en
    ``[max(ret_home_start, rd), min(ret_home_end, rd + slack)]`` ∩ ``[od+min, od+max]``.
    """
    ha_lo = max(ret_home_start, rd)
    ha_hi = min(ret_home_end, rd + timedelta(days=home_slack_days))
    if ha_lo > ha_hi:
        return False
    trip_lo = od + timedelta(days=min_trip_days)
    trip_hi = od + timedelta(days=max_trip_days)
    overlap_lo = max(ha_lo, trip_lo)
    overlap_hi = min(ha_hi, trip_hi)
    return overlap_lo <= overlap_hi


def build_search_pairs(
    outbound_start: date,
    outbound_end: date,
    min_trip_days: int,
    max_trip_days: int,
    ret_home_start: date,
    ret_home_end: date,
    home_slack_days: int = PAIR_HOME_ARRIVAL_SLACK_DAYS,
) -> tuple[list[tuple[date, date]], int]:
    """Return (pairs, raw_count_before_filter).

    Genera cada ``(od, od+span)`` con ``span`` en ``[min_trip_days, max_trip_days]`` (fechas de query a la API),
    y **elimina** los que no pueden cumplir a la vez ventana de llegada a casa y largo de viaje en días de calendario.
    """
    raw_pairs: list[tuple[date, date]] = []
    for od in daterange(outbound_start, outbound_end):
        for span in range(min_trip_days, max_trip_days + 1):
            rd = od + timedelta(days=span)
            raw_pairs.append((od, rd))

    uniq_raw = list(dict.fromkeys(raw_pairs))
    n_uniq_before_filter = len(uniq_raw)
    pairs = [
        (od, rd)
        for od, rd in uniq_raw
        if pair_plausible_for_home_and_trip_length(
            od,
            rd,
            ret_home_start,
            ret_home_end,
            min_trip_days,
            max_trip_days,
            home_slack_days,
        )
    ]
    dropped = len(uniq_raw) - len(pairs)
    if dropped:
        log.info(
            "Pre-filtro (ventana llegada a casa + MIN/MAX días de viaje): %d pares descartados, %d quedan.",
            dropped,
            len(pairs),
        )
    if len(pairs) > MAX_SEARCH_PAIRS:
        log.warning(
            "Truncating filtered pairs from %d to MAX_SEARCH_PAIRS=%d (raise constant or narrow windows).",
            len(pairs),
            MAX_SEARCH_PAIRS,
        )
        pairs = pairs[:MAX_SEARCH_PAIRS]
    return pairs, n_uniq_before_filter


def build_pairs_dest_return_dates(
    out_start: date,
    out_end: date,
    ret_start: date,
    ret_end: date,
    max_pairs: int,
    max_span_days: int,
) -> tuple[list[tuple[date, date]], int]:
    """Cartesiano: cada salida de ida (origen) × cada regreso desde destino; requiere ``od <= rd`` y span razonable."""
    raw_pairs: list[tuple[date, date]] = []
    for od in daterange(out_start, out_end):
        for rd in daterange(ret_start, ret_end):
            if od <= rd and (rd - od).days <= max_span_days:
                raw_pairs.append((od, rd))
    uniq_raw = list(dict.fromkeys(raw_pairs))
    n_before = len(uniq_raw)
    pairs = uniq_raw
    if len(pairs) > max_pairs:
        log.warning(
            "Truncating DEST_RETURN_DATES pairs from %d to MAX_SEARCH_PAIRS=%d.",
            len(pairs),
            max_pairs,
        )
        pairs = pairs[:max_pairs]
    return pairs, n_before


if SEARCH_MODE == "VACATION_WINDOW":
    PAIRS, _N_PAIRS_BEFORE_PRE_FILTER = build_search_pairs(
        OUTBOUND_DEPART_START,
        OUTBOUND_DEPART_END,
        MIN_TRIP_LENGTH_DAYS,
        MAX_TRIP_LENGTH_DAYS,
        RETURN_ARRIVAL_HOME_START,
        RETURN_ARRIVAL_HOME_END,
        PAIR_HOME_ARRIVAL_SLACK_DAYS,
    )
else:
    PAIRS, _N_PAIRS_BEFORE_PRE_FILTER = build_pairs_dest_return_dates(
        OUTBOUND_DEPART_START,
        OUTBOUND_DEPART_END,
        RETURN_DEPART_DEST_START,
        RETURN_DEPART_DEST_END,
        MAX_SEARCH_PAIRS,
        MAX_OD_RD_SPAN_DAYS,
    )

if not PAIRS:
    raise ValueError(
        "No quedó ningún par (ida, regreso). Revisá fechas, SEARCH_MODE, o ampliá ventanas / MAX_SEARCH_PAIRS en constantes."
    )

_approx_posts = len(PAIRS) * (1 + RT_RECURSION_TOP_N)
_out_days = (OUTBOUND_DEPART_END - OUTBOUND_DEPART_START).days + 1
log.info(
    "SEARCH_MODE=%s: %d (outbound, return_depart) pairs (raw/pre-filter count=%d); "
    "outbound-day span=%d; home window %s..%s; ~%d POSTs (~1+%d per pair).",
    SEARCH_MODE,
    len(PAIRS),
    _N_PAIRS_BEFORE_PRE_FILTER,
    _out_days,
    RETURN_ARRIVAL_HOME_START.isoformat(),
    RETURN_ARRIVAL_HOME_END.isoformat(),
    _approx_posts,
    RT_RECURSION_TOP_N,
)

# Vista previa explícita (antes de cualquier llamada a la API)
_preview_rows = []
for _i, (_od, _rd) in enumerate(PAIRS, start=1):
    _ha_lo = max(RETURN_ARRIVAL_HOME_START, _rd)
    _ha_hi = min(RETURN_ARRIVAL_HOME_END, _rd + timedelta(days=PAIR_HOME_ARRIVAL_SLACK_DAYS))
    _trip_lo = _od + timedelta(days=MIN_TRIP_LENGTH_DAYS)
    _trip_hi = _od + timedelta(days=MAX_TRIP_LENGTH_DAYS)
    _ol = max(_ha_lo, _trip_lo)
    _oh = min(_ha_hi, _trip_hi)
    _preview_rows.append(
        Row(
            n=_i,
            search_mode=SEARCH_MODE,
            desde_ida_origen=_od.isoformat(),
            hasta_regreso_desde_dest=_rd.isoformat(),
            duracion_calendario_dias=(_rd - _od).days,
            llegada_casa_plausible_desde=_ol.isoformat(),
            llegada_casa_plausible_hasta=_oh.isoformat(),
        )
    )
if SEARCH_MODE == "VACATION_WINDOW":
    _plan_extra = (
        f"Vacaciones: primera salida {OUTBOUND_DEPART_START}, última llegada a casa {RETURN_ARRIVAL_HOME_END}; "
        f"duración permitida {MIN_TRIP_LENGTH_DAYS}…{MAX_TRIP_LENGTH_DAYS} días."
    )
else:
    _plan_extra = (
        f"Fechas fijas: ida desde origen {OUTBOUND_DEPART_START}…{OUTBOUND_DEPART_END}, "
        f"regreso desde destino {RETURN_DEPART_DEST_START}…{RETURN_DEPART_DEST_END} "
        f"(cartesiano; span máx. {MAX_OD_RD_SPAN_DAYS} días)."
    )
print(
    f"\nPlan de búsqueda ({SEARCH_MODE}): {len(PAIRS)} llamada(s) a la API (una por fila). "
    f"{_plan_extra}\n",
    flush=True,
)
display(spark.createDataFrame(_preview_rows))  # noqa: F821

# COMMAND ----------
# --- Helpers: extract datetimes, filter results ---


def _last_leg(fr: FlightResult) -> Any:
    return fr.legs[-1]


def home_arrival_date(fr: FlightResult, home: Airport) -> date | None:
    leg = _last_leg(fr)
    if leg.arrival_airport != home:
        return None
    return leg.arrival_datetime.date()


def first_outbound_departure_date(fr: FlightResult, origin: Airport) -> date | None:
    if not fr.legs:
        return None
    if fr.legs[0].departure_airport != origin:
        return None
    return fr.legs[0].departure_datetime.date()


def trip_length_days(outbound: FlightResult, ret: FlightResult) -> int | None:
    o = first_outbound_departure_date(outbound, AIRPORT_ORIGIN)
    h = home_arrival_date(ret, AIRPORT_ORIGIN)
    if o is None or h is None:
        return None
    return (h - o).days


def passes_post_filters(
    outbound: FlightResult,
    ret: FlightResult,
    ret_home_start: date,
    ret_home_end: date,
    min_len: int,
    max_len: int,
) -> bool:
    tl = trip_length_days(outbound, ret)
    if tl is None or tl < min_len or tl > max_len:
        return False
    ha = home_arrival_date(ret, AIRPORT_ORIGIN)
    if ha is None or ha < ret_home_start or ha > ret_home_end:
        return False
    return True


def _date_and_time(dt: datetime) -> tuple[str, str]:
    """Local calendar date and clock time (no TZ offset in string)."""
    return dt.date().isoformat(), dt.strftime("%H:%M")


def _legs_airline_flight_summary(legs: list[Any]) -> str:
    """e.g. ``B6 123 / B6 456`` for connections."""
    parts: list[str] = []
    for leg in legs:
        parts.append(f"{leg.airline.value} {leg.flight_number}".strip())
    return " / ".join(parts)


def flatten_roundtrip(
    outbound: FlightResult,
    ret: FlightResult,
    od: date,
    rd: date,
) -> dict[str, Any]:
    out0, out1 = outbound.legs[0], outbound.legs[-1]
    r0, r1 = ret.legs[0], ret.legs[-1]
    o_dep_d, o_dep_t = _date_and_time(out0.departure_datetime)
    o_arr_d, o_arr_t = _date_and_time(out1.arrival_datetime)
    r_dep_d, r_dep_t = _date_and_time(r0.departure_datetime)
    r_arr_d, r_arr_t = _date_and_time(r1.arrival_datetime)
    return {
        "origin_iata": AIRPORT_ORIGIN.name,
        "destination_iata": AIRPORT_DEST.name,
        "query_outbound_date": od.isoformat(),
        "query_return_depart_date": rd.isoformat(),
        "total_price": float(outbound.price),
        "currency": outbound.currency,
        "outbound_stops": outbound.stops,
        "return_stops": ret.stops,
        "outbound_airlines_flights": _legs_airline_flight_summary(outbound.legs),
        "return_airlines_flights": _legs_airline_flight_summary(ret.legs),
        "outbound_dep_airport": out0.departure_airport.value,
        "outbound_arr_airport": out1.arrival_airport.value,
        "outbound_dep_date": o_dep_d,
        "outbound_dep_time": o_dep_t,
        "outbound_arr_date": o_arr_d,
        "outbound_arr_time": o_arr_t,
        "return_dep_airport": r0.departure_airport.value,
        "return_arr_airport": r1.arrival_airport.value,
        "return_dep_date": r_dep_d,
        "return_dep_time": r_dep_t,
        "return_arr_date": r_arr_d,
        "return_arr_time": r_arr_t,
        "trip_length_days": trip_length_days(outbound, ret),
        "outbound_duration_min": int(outbound.duration),
        "return_duration_min": int(ret.duration),
        "outbound_leg_count": len(outbound.legs),
        "return_leg_count": len(ret.legs),
        "matches_trip_constraints": passes_post_filters(
            outbound,
            ret,
            RETURN_ARRIVAL_HOME_START,
            RETURN_ARRIVAL_HOME_END,
            MIN_TRIP_LENGTH_DAYS,
            MAX_TRIP_LENGTH_DAYS,
        ),
    }


# Columns that define the same trip for de-duplication (exclude price so we keep cheapest).
_DEDUPE_PARTITION_COLS = [
    "outbound_dep_date",
    "outbound_dep_time",
    "outbound_arr_date",
    "outbound_arr_time",
    "return_dep_date",
    "return_dep_time",
    "return_arr_date",
    "return_arr_time",
    "outbound_airlines_flights",
    "return_airlines_flights",
    "outbound_dep_airport",
    "outbound_arr_airport",
    "return_dep_airport",
    "return_arr_airport",
]


# COMMAND ----------
# --- Search with retries + jittered pacing ---


@dataclass
class SearchStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    rate_limited_skips: int = 0


def _is_rate_limited(exc: BaseException) -> bool:
    """Detect HTTP 429 / rate-limit across urllib, httpx, or wrapped fli errors."""
    if "429" in str(exc):
        return True
    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, HTTPError) and getattr(cur, "code", None) == 429:
            return True
        resp = getattr(cur, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 429:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def sleep_after_pair() -> None:
    """Pausa entre pares; escala hacia abajo si hay pocos pares (evita ~50s × N cuando N es pequeño)."""
    n = max(1, len(PAIRS))
    if n <= 8:
        scale = 0.35
    elif n < 25:
        scale = 0.65
    else:
        scale = 1.0
    lo = max(0.5, MIN_DELAY_SEC * scale)
    hi = max(lo + 0.5, MAX_DELAY_SEC * scale)
    time.sleep(random.uniform(lo, hi))


def sort_roundtrips_by_total_price(
    pairs: list[tuple[FlightResult, FlightResult]],
) -> list[tuple[FlightResult, FlightResult]]:
    """Orden estable por precio total (``outbound.price`` en RT) y duración."""
    return sorted(
        pairs,
        key=lambda pr: (float(pr[0].price), int(pr[0].duration), int(pr[1].duration)),
    )


def take_top_cheapest_roundtrips(
    pairs: list[tuple[FlightResult, FlightResult]],
    n: int,
) -> list[tuple[FlightResult, FlightResult]]:
    """Ordena por precio total y devuelve como máximo ``n`` opciones (para volcar muchas filas al DataFrame)."""
    ordered = sort_roundtrips_by_total_price(pairs)
    if n <= 0:
        return ordered
    return ordered[:n]


def run_one_search(out_d: date, ret_d: date, searcher: SearchFlights) -> list[tuple[FlightResult, FlightResult]]:
    filters = FlightSearchFilters(
        trip_type=TripType.ROUND_TRIP,
        passenger_info=PassengerInfo(adults=ADULTS),
        show_all_results=SHOW_ALL_API_RESULTS,
        flight_segments=[
            FlightSegment(
                departure_airport=[[AIRPORT_ORIGIN, 0]],
                arrival_airport=[[AIRPORT_DEST, 0]],
                travel_date=out_d.isoformat(),
            ),
            FlightSegment(
                departure_airport=[[AIRPORT_DEST, 0]],
                arrival_airport=[[AIRPORT_ORIGIN, 0]],
                travel_date=ret_d.isoformat(),
            ),
        ],
        stops=MAX_STOPS,
        seat_type=SEAT_TYPE,
        sort_by=SORT_BY,
        exclude_basic_economy=EXCLUDE_BASIC_ECONOMY,
        bags=BagsFilter(checked_bags=0, carry_on=True),
    )
    res = searcher.search(filters, top_n=RT_RECURSION_TOP_N)
    if res is None:
        return []
    return list(res)


def search_with_retries(
    out_d: date, ret_d: date, searcher: SearchFlights
) -> tuple[list[tuple[FlightResult, FlightResult]], str | None]:
    """Return (results, error_message). On total failure error_message is set; never raises."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_one_search(out_d, ret_d, searcher), None
        except Exception as exc:  # noqa: BLE001 — vendor/network
            last_exc = exc
            if _is_rate_limited(exc):
                backoff = min(180.0, 12.0 * (1.75 ** (attempt - 1))) + random.uniform(0, 6)
            else:
                backoff = min(60.0, 2.5**attempt) + random.uniform(0, 1.0)
            print(
                f"  Reintento {attempt}/{MAX_RETRIES}: error ({type(exc).__name__}): {str(exc)[:400]} — espera {backoff:.1f}s",
                flush=True,
            )
            log.warning(
                "Search failed od=%s ret=%s attempt=%d/%d: %s (sleep %.1fs)",
                out_d,
                ret_d,
                attempt,
                MAX_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)
    err = repr(last_exc) if last_exc else "unknown error"
    return [], err


stats = SearchStats()
searcher = SearchFlights()
rows: list[Row] = []
failure_rows: list[Row] = []

for idx, (od, rd) in enumerate(PAIRS, start=1):
    span = (rd - od).days
    print(
        f"\n{'=' * 72}\n"
        f"BÚSQUEDA {idx}/{len(PAIRS)}  |  {AIRPORT_ORIGIN.name} → {AIRPORT_DEST.name}  (ida y vuelta)\n"
        f"  · Salida de ida desde {AIRPORT_ORIGIN.name} (fecha en la query):  {od.isoformat()}\n"
        f"  · Vuelo de regreso **desde** {AIRPORT_DEST.name} (fecha en la query): {rd.isoformat()}\n"
        f"  · Días calendario entre esas dos fechas de trayecto: {span}"
        + (
            f" (MIN/MAX trip: {MIN_TRIP_LENGTH_DAYS}…{MAX_TRIP_LENGTH_DAYS})\n"
            if SEARCH_MODE == "VACATION_WINDOW"
            else " (modo fechas: acotá duración o aerolínea en Spark si hace falta)\n"
        )
        f"{'=' * 72}",
        flush=True,
    )
    stats.attempted += 1
    pairs, err = search_with_retries(od, rd, searcher)
    if err:
        stats.failed += 1
        rate_hit = "429" in err
        if rate_hit:
            stats.rate_limited_skips += 1
        failure_rows.append(
            Row(
                query_outbound_date=od.isoformat(),
                query_return_depart_date=rd.isoformat(),
                error=err[:2000],
            )
        )
        print(f"  ✖ Sin resultados tras reintentos: {str(err)[:500]}", flush=True)
        log.error("Skipped od=%s ret=%s after retries: %s", od, rd, err)
        if rate_hit:
            cool = RATE_LIMIT_COOLDOWN_SEC + random.uniform(0, 15)
            print(f"  ⏳ Cool-down 429: durmiendo {cool:.1f}s antes del siguiente par.", flush=True)
            log.warning("Rate limit cool-down sleeping %.1fs before next query", cool)
            time.sleep(cool)
        sleep_after_pair()
        continue

    stats.succeeded += 1
    raw_n = len(pairs)
    pairs = take_top_cheapest_roundtrips(pairs, TOP_RESULTS_PER_SEARCH)
    after_top = len(pairs)
    kept = 0
    for outbound, ret in pairs:
        if APPLY_TRIP_WINDOW_FILTERS and not passes_post_filters(
            outbound,
            ret,
            RETURN_ARRIVAL_HOME_START,
            RETURN_ARRIVAL_HOME_END,
            MIN_TRIP_LENGTH_DAYS,
            MAX_TRIP_LENGTH_DAYS,
        ):
            continue
        rec = flatten_roundtrip(outbound, ret, od, rd)
        rows.append(Row(**rec))
        kept += 1
    print(
        f"  ✓ OK: {raw_n} RT devueltos por la API → "
        f"tras TOP_RESULTS={TOP_RESULTS_PER_SEARCH}: {after_top} filas → "
        f"{'filtradas por ventana' if APPLY_TRIP_WINDOW_FILTERS else 'sin filtro in-notebook'} → "
        f"{kept} fila(s) añadidas al resultado.\n",
        flush=True,
    )
    log.debug(
        "od=%s rd=%s api_pairs=%d after_top=%d rows=%d apply_trip_filters=%s",
        od,
        rd,
        raw_n,
        after_top,
        kept,
        APPLY_TRIP_WINDOW_FILTERS,
    )
    sleep_after_pair()

log.info(
    "Done: attempted=%s succeeded=%s failed=%s rate_limited=%s rows=%s failures_logged=%s",
    stats.attempted,
    stats.succeeded,
    stats.failed,
    stats.rate_limited_skips,
    len(rows),
    len(failure_rows),
)

# COMMAND ----------
# --- Spark output ---

if not rows:
    log.warning(
        "No rows — widen date grids, set APPLY_TRIP_WINDOW_FILTERS=false, or relax MAX_STOPS / Basic Economy."
    )
else:
    df = spark.createDataFrame(rows)  # noqa: F821
    before = df.count()
    part = Window.partitionBy(*_DEDUPE_PARTITION_COLS).orderBy(
        F.col("total_price").asc(),
        F.col("query_outbound_date").asc(),
        F.col("query_return_depart_date").asc(),
    )
    df = (
        df.withColumn("_dedupe_rank", F.row_number().over(part))
        .filter(F.col("_dedupe_rank") == 1)
        .drop("_dedupe_rank")
    )
    after = df.count()
    if before > after:
        log.info("Removed %d duplicate itinerary row(s); %d unique remain.", before - after, after)
    display_cols = [
        "origin_iata",
        "destination_iata",
        "query_outbound_date",
        "query_return_depart_date",
        "total_price",
        "currency",
        "outbound_stops",
        "return_stops",
        "outbound_airlines_flights",
        "return_airlines_flights",
        "outbound_dep_airport",
        "outbound_dep_date",
        "outbound_dep_time",
        "outbound_arr_airport",
        "outbound_arr_date",
        "outbound_arr_time",
        "return_dep_airport",
        "return_dep_date",
        "return_dep_time",
        "return_arr_airport",
        "return_arr_date",
        "return_arr_time",
        "trip_length_days",
        "outbound_duration_min",
        "return_duration_min",
        "outbound_leg_count",
        "return_leg_count",
        "matches_trip_constraints",
    ]
    df = df.select(*display_cols).orderBy(F.col("total_price").asc())
    display(df)  # noqa: F821

if failure_rows:
    log.warning("Showing %d failed (outbound, return) queries — inspect errors (often 429 rate limits).", len(failure_rows))
    display(spark.createDataFrame(failure_rows))  # noqa: F821

# COMMAND ----------
# MAGIC %md
# MAGIC ### Widgets (solo lo operativo)
# MAGIC - **`SEARCH_MODE`:** `VACATION_WINDOW` | `DEST_RETURN_DATES`.
# MAGIC - **`VACATION_WINDOW`:** `VACATION_FIRST_DAY`, `VACATION_LAST_HOME_DAY` (**YYYYMMDD**), `MIN_TRIP_DAYS`, `MAX_TRIP_DAYS`. Primera salida posible = primera fecha; última **llegada a casa** = última fecha; se generan salidas de ida hasta poder cumplir duración mínima y llegar a tiempo.
# MAGIC - **`DEST_RETURN_DATES`:** `OUTBOUND_DEPART_START`/`END`, `RETURN_DEPART_DEST_START`/`END` (**YYYYMMDD**). Producto cartesiano (cada ida × cada regreso desde destino) con `od <= rd` y span máx. fijo en código (`MAX_OD_RD_SPAN_DAYS`).
# MAGIC - **Aeropuertos:** `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT` (IATA 3 letras, enum `fli.models.Airport`).
# MAGIC - **Cabina / orden:** `SEAT_TYPE`, `SORT_BY` (recomendado `BEST` para no sesgar todo a ULCC), `EXCLUDE_BASIC_ECONOMY`, `ADULTS`.
# MAGIC - **`APPLY_TRIP_WINDOW_FILTERS`:** si `true`, filtra filas in-notebook; si `false`, usá Spark (p. ej. excluir Frontier con `~F.col(...).contains("Frontier")`).
# MAGIC
# MAGIC ### Constantes (celda “Tunables”, sin widget)
# MAGIC `MAX_SEARCH_PAIRS`, `TOP_RESULTS_PER_SEARCH`, `RT_RECURSION_TOP_N`, `SHOW_ALL_API_RESULTS`, delays, retries,
# MAGIC `RATE_LIMIT_COOLDOWN_SEC`, `MAX_OD_RD_SPAN_DAYS`, `PAIR_HOME_ARRIVAL_SLACK_DAYS`, `MAX_STOPS`.
# MAGIC
# MAGIC Primera ejecución crea widgets; cambiá valores y **Run all** desde la celda de lectura/validación hacia abajo.
