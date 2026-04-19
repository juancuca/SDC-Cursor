# Databricks notebook source
# MAGIC %md
# MAGIC # Round-trip flight grid (carry-on–aware pricing)
# MAGIC
# MAGIC Searches many **(outbound departure date, return departure date)** combinations using
# MAGIC [`flights` / `fli`](https://github.com/punitarani/fli) (unofficial Google Flights API client).
# MAGIC
# MAGIC **Resultados por búsqueda:** para cada **par (fecha ida en origen, fecha vuelo de regreso desde destino)** se llama
# MAGIC a la API. Antes se **descartan** pares imposibles: `MIN`/`MAX_TRIP_LENGTH_DAYS` se interpretan como días de calendario
# MAGIC entre la salida de ida y una **llegada a casa** plausible dentro de `RETURN_ARRIVAL_HOME_*` (cruce con `rd`…`rd+2`
# MAGIC días por vuelos que llegan tarde). Así un rango largo de salidas **no** multiplica búsquedas que nunca podrían cumplir
# MAGIC “solo 10 días de viaje” y llegada a casa en tu ventana. **Un solo día de salida** con `MIN`…`MAX` en **7–10** y ventana
# MAGIC de regreso estrecha puede quedar en **0–4** pares según solape, no siempre 4. Cada par ejecuta la recursión RT de **fli**:
# MAGIC **≈ 1 + `RT_RECURSION_TOP_N` POST** a Google (ver `fli/search/flights.py`: `search(..., top_n=…)` expande idas y luego
# MAGIC vuelta por rama). **`TOP_RESULTS_PER_SEARCH`** solo limita filas **después** de ordenar; no reduce esos POSTs.
# MAGIC `SHOW_ALL_API_RESULTS=true` agranda la primera respuesta (más lento / 429). Entre pares hay pausa configurable
# MAGIC (`MIN_DELAY_SEC`…`MAX_DELAY_SEC`), escalada hacia abajo si hay pocos pares.
# MAGIC
# MAGIC **Carry-on:** `BagsFilter(carry_on=True)` makes Google Flights **quote prices including a carry-on** so
# MAGIC comparisons match what you pay if you need an overhead bag. It does **not** guarantee every fare
# MAGIC includes a *complimentary* carry-on; pairing with **`exclude_basic_economy=True`** avoids many U.S.
# MAGIC **Basic Economy** fares that often have no free carry-on.
# MAGIC
# MAGIC **Compliance:** Unofficial API; respect Google’s terms, rate limits, and your org policy. Use only
# MAGIC on clusters with appropriate egress. Not legal advice.
# MAGIC
# MAGIC **Install note:** On PyPI the package is versioned **0.x** (e.g. `0.8.4`), not 2.x. Requires **Python ≥ 3.10**
# MAGIC (matches current `flights` wheels; Databricks serverless is fine if the runtime is 3.10+).
# MAGIC
# MAGIC **`typing_extensions`:** Newer `pydantic` / `fli` stacks expect **`Sentinel`** in `typing_extensions`. Databricks still
# MAGIC ships an older copy under `/databricks/python/...` until you upgrade and **restart Python**, or imports can fail.

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
    t0 = date.today()
    _w_create_text("ORIGIN_AIRPORT", "TPA")
    _w_create_text("DESTINATION_AIRPORT", "SFO")
    _w_create_text("OUTBOUND_DEPART_START", "20260614")
    _w_create_text("OUTBOUND_DEPART_END", "20260614")
    _w_create_text("RETURN_ARRIVAL_HOME_START", "20260621")
    _w_create_text("RETURN_ARRIVAL_HOME_END", "20260621")
    _w_create_dropdown("MIN_TRIP_LENGTH_DAYS", "7", [str(i) for i in range(1, 22)])
    _w_create_dropdown("MAX_TRIP_LENGTH_DAYS", "10", [str(i) for i in range(1, 31)])
    _w_create_dropdown(
        "MAX_API_CALLS",
        "9999",
        [str(i) for i in (20, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 9999)],
    )
    _w_create_text("MIN_DELAY_SEC", "6")
    _w_create_text("MAX_DELAY_SEC", "50")
    _w_create_dropdown("MAX_RETRIES", "10", [str(i) for i in (3, 4, 5, 6, 8, 10)])
    _w_create_dropdown("RATE_LIMIT_COOLDOWN_SEC", "75", [str(i) for i in (30, 45, 60, 75, 90, 120)])
    _w_create_dropdown("ADULTS", "1", [str(i) for i in range(1, 7)])
    _w_create_dropdown(
        "SEAT_TYPE",
        "ECONOMY",
        ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"],
    )
    _w_create_dropdown(
        "MAX_STOPS",
        "TWO_OR_FEWER_STOPS",
        ["ANY", "NON_STOP", "ONE_STOP_OR_FEWER", "TWO_OR_FEWER_STOPS"],
    )
    _w_create_dropdown(
        "SORT_BY",
        "CHEAPEST",
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
    _w_create_dropdown(
        "TOP_RESULTS_PER_SEARCH",
        "30",
        [str(i) for i in (10, 15, 20, 25, 30, 35, 40, 50)],
    )
    _w_create_dropdown("SHOW_ALL_API_RESULTS", "true", ["false", "true"])
    _w_create_dropdown("APPLY_TRIP_WINDOW_FILTERS", "false", ["false", "true"])
    _w_create_dropdown(
        "RT_RECURSION_TOP_N",
        "8",
        [str(i) for i in (3, 5, 6, 8, 10, 12, 15)],
    )
    try:
        dbutils.widgets.remove("PRICE_CUTOFF_PERCENT")  # noqa: F821
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


def _w_float(name: str) -> float:
    return float(_w_get(name))


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

# Outbound: depart **origin** on these dates (inclusive), widgets use YYYYMMDD
OUTBOUND_DEPART_START = parse_ymd("OUTBOUND_DEPART_START")
OUTBOUND_DEPART_END = parse_ymd("OUTBOUND_DEPART_END")

# Return: **arrival back at origin (home)** calendar date in this inclusive window (YYYYMMDD)
RETURN_ARRIVAL_HOME_START = parse_ymd("RETURN_ARRIVAL_HOME_START")
RETURN_ARRIVAL_HOME_END = parse_ymd("RETURN_ARRIVAL_HOME_END")

MIN_TRIP_LENGTH_DAYS = _w_int("MIN_TRIP_LENGTH_DAYS")
MAX_TRIP_LENGTH_DAYS = _w_int("MAX_TRIP_LENGTH_DAYS")
MAX_API_CALLS = _w_int("MAX_API_CALLS")
MIN_DELAY_SEC = _w_float("MIN_DELAY_SEC")
MAX_DELAY_SEC = _w_float("MAX_DELAY_SEC")
MAX_RETRIES = _w_int("MAX_RETRIES")
RATE_LIMIT_COOLDOWN_SEC = _w_float("RATE_LIMIT_COOLDOWN_SEC")

ADULTS = _w_int("ADULTS")
SEAT_TYPE = getattr(SeatType, _w_get("SEAT_TYPE"))
MAX_STOPS = getattr(MaxStops, _w_get("MAX_STOPS"))
SORT_BY = getattr(SortBy, _w_get("SORT_BY"))
EXCLUDE_BASIC_ECONOMY = _w_get("EXCLUDE_BASIC_ECONOMY").lower() in ("1", "true", "yes")
TOP_RESULTS_PER_SEARCH = _w_int("TOP_RESULTS_PER_SEARCH")
SHOW_ALL_API_RESULTS = _w_get("SHOW_ALL_API_RESULTS").lower() in ("1", "true", "yes")
APPLY_TRIP_WINDOW_FILTERS = _w_get("APPLY_TRIP_WINDOW_FILTERS").lower() in ("1", "true", "yes")
RT_RECURSION_TOP_N = _w_int("RT_RECURSION_TOP_N")

# COMMAND ----------
# --- Validation (fail fast) ---

if AIRPORT_ORIGIN == AIRPORT_DEST:
    raise ValueError("ORIGIN_AIRPORT and DESTINATION_AIRPORT must be different")

if OUTBOUND_DEPART_END < OUTBOUND_DEPART_START:
    raise ValueError("OUTBOUND_DEPART_END must be on or after OUTBOUND_DEPART_START")
if RETURN_ARRIVAL_HOME_END < RETURN_ARRIVAL_HOME_START:
    raise ValueError("RETURN_ARRIVAL_HOME_END must be on or after RETURN_ARRIVAL_HOME_START")
if MIN_TRIP_LENGTH_DAYS < 1 or MAX_TRIP_LENGTH_DAYS < MIN_TRIP_LENGTH_DAYS:
    raise ValueError("MIN_TRIP_LENGTH_DAYS / MAX_TRIP_LENGTH_DAYS are invalid")
if MAX_API_CALLS < 1:
    raise ValueError("MAX_API_CALLS must be >= 1")
if MIN_DELAY_SEC <= 0 or MAX_DELAY_SEC <= 0 or MAX_DELAY_SEC < MIN_DELAY_SEC:
    raise ValueError("MIN_DELAY_SEC / MAX_DELAY_SEC must be positive and MIN <= MAX")
if MAX_RETRIES < 1:
    raise ValueError("MAX_RETRIES must be >= 1")
if RATE_LIMIT_COOLDOWN_SEC < 0:
    raise ValueError("RATE_LIMIT_COOLDOWN_SEC must be >= 0")
if TOP_RESULTS_PER_SEARCH < 1 or TOP_RESULTS_PER_SEARCH > 200:
    raise ValueError("TOP_RESULTS_PER_SEARCH must be between 1 and 200")
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

if MAX_API_CALLS > 500:
    log.warning("MAX_API_CALLS=%d is very high; consider lowering to reduce runtime.", MAX_API_CALLS)

if SHOW_ALL_API_RESULTS and TOP_RESULTS_PER_SEARCH > 50:
    log.warning(
        "SHOW_ALL_API_RESULTS=true with TOP_RESULTS_PER_SEARCH=%d may slow searches and increase 429s.",
        TOP_RESULTS_PER_SEARCH,
    )

# COMMAND ----------
# --- Candidate (outbound_date, return_depart_date) pairs ---

# Días extra tras `return_depart_dest` para modelar “llegada a casa” (vuelos largos / conexiones).
PAIR_HOME_ARRIVAL_SLACK_DAYS = 2


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
    if len(pairs) > MAX_API_CALLS:
        log.warning(
            "Truncating filtered pairs from %d to MAX_API_CALLS=%d (raise cap or narrow windows).",
            len(pairs),
            MAX_API_CALLS,
        )
        pairs = pairs[:MAX_API_CALLS]
    return pairs, n_uniq_before_filter


PAIRS, _N_PAIRS_BEFORE_PRE_FILTER = build_search_pairs(
    OUTBOUND_DEPART_START,
    OUTBOUND_DEPART_END,
    MIN_TRIP_LENGTH_DAYS,
    MAX_TRIP_LENGTH_DAYS,
    RETURN_ARRIVAL_HOME_START,
    RETURN_ARRIVAL_HOME_END,
    PAIR_HOME_ARRIVAL_SLACK_DAYS,
)
if not PAIRS:
    raise ValueError(
        "No quedó ningún par (ida, regreso) tras el pre-filtro. Ensancha RETURN_ARRIVAL_HOME_* o MIN/MAX_TRIP_LENGTH_DAYS, "
        "o acorta el rango de salida de ida."
    )

_approx_posts = len(PAIRS) * (1 + RT_RECURSION_TOP_N)
_out_days = (OUTBOUND_DEPART_END - OUTBOUND_DEPART_START).days + 1
_span_count = MAX_TRIP_LENGTH_DAYS - MIN_TRIP_LENGTH_DAYS + 1
log.info(
    "Prepared %d (outbound, return_depart) pairs (%d únicos antes del pre-filtro) — ≈ %d outbound day(s) × %d span(s) "
    "en grilla; MIN..MAX_TRIP_LENGTH_DAYS=%d..%d; ventana llegada a casa %s..%s; cada par ≈ 1 + %d POSTs. "
    "Cota ≈ %d POSTs.",
    len(PAIRS),
    _N_PAIRS_BEFORE_PRE_FILTER,
    _out_days,
    _span_count,
    MIN_TRIP_LENGTH_DAYS,
    MAX_TRIP_LENGTH_DAYS,
    RETURN_ARRIVAL_HOME_START.isoformat(),
    RETURN_ARRIVAL_HOME_END.isoformat(),
    RT_RECURSION_TOP_N,
    _approx_posts,
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
            desde_ida_origen=_od.isoformat(),
            hasta_regreso_desde_dest=_rd.isoformat(),
            duracion_calendario_dias=(_rd - _od).days,
            llegada_casa_plausible_desde=_ol.isoformat(),
            llegada_casa_plausible_hasta=_oh.isoformat(),
        )
    )
print(
    f"\nPlan de búsqueda: {len(PAIRS)} llamada(s) a la API (una por fila). "
    f"Ventana llegada a casa: {RETURN_ARRIVAL_HOME_START} … {RETURN_ARRIVAL_HOME_END}. "
    f"Duración viaje (días calendario salida ida → llegada casa): {MIN_TRIP_LENGTH_DAYS}…{MAX_TRIP_LENGTH_DAYS}.\n",
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
        f"  · Días calendario entre esas dos fechas de trayecto: {span} "
        f"(MIN/MAX trip permitidos: {MIN_TRIP_LENGTH_DAYS}…{MAX_TRIP_LENGTH_DAYS})\n"
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
# MAGIC ### Parameters (widgets)
# MAGIC - **Airports:** **text** `ORIGIN_AIRPORT` and `DESTINATION_AIRPORT` — **3-letter IATA** (e.g. `TPA`, `LAX`). Must exist in `fli.models.Airport`.
# MAGIC - **Volume:** **dropdown** `TOP_RESULTS_PER_SEARCH` (p. ej. 30) = N opciones **más baratas** **después** de la API (orden por precio); **no** controla cuántos POST hace Google. **dropdown** `RT_RECURSION_TOP_N` = argumento `top_n` de `fli` en RT (**≈ 1 + top_n POSTs por par de fechas**); subirlo mucho multiplica llamadas. **dropdown** `SHOW_ALL_API_RESULTS` (`false` ≈ ~30 filas API, `true` = más resultados, más lento/429). **dropdown** `APPLY_TRIP_WINDOW_FILTERS`: si `false`, se vuelcan las filas y se usa **`matches_trip_constraints`** + fechas en Spark.
# MAGIC - **Dates** (`OUTBOUND_DEPART_*`, `RETURN_ARRIVAL_HOME_*`): **text** widgets, **`YYYYMMDD`**. La ventana **llegada a casa** acota los pares **antes** de la API (junto con MIN/MAX días de viaje); ver tabla “Plan de búsqueda” y columnas `llegada_casa_plausible_*`.
# MAGIC - **Trip length / caps**: **dropdowns** for `MIN_TRIP_LENGTH_DAYS`, `MAX_TRIP_LENGTH_DAYS`, `MAX_API_CALLS`, `MAX_RETRIES`, `ADULTS`.
# MAGIC - **Delays**: **text** `MIN_DELAY_SEC`, `MAX_DELAY_SEC` (segundos, jitter **entre cada par**; si hay ≤8 pares se escala ~×0.35 para no multiplicar 50s × pocos intentos).
# MAGIC - **429 cool-down**: **dropdown** `RATE_LIMIT_COOLDOWN_SEC` — extra sleep after a fully failed query whose error mentions `429`.
# MAGIC - **Search**: **dropdowns** `SEAT_TYPE`, `MAX_STOPS`, `SORT_BY`, `EXCLUDE_BASIC_ECONOMY` (`true` / `false`).
# MAGIC Widgets are created automatically on first run; adjust values in the UI then **Run all** from the read/validation cell downward.
