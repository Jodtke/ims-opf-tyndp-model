"""Network reduction and topology helpers for the maintenance OPF workflow.

The functions in this module translate raw European grid data into the compact
AC/DC network representation used by the stochastic maintenance model. The
reduced network keeps the quantities that matter for the optimization:
geographic bus positions, country membership, aggregated AC corridors, DC links,
parallel circuit counts, transfer capacities, and reactance-derived DC
susceptances.
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Tuple


def _opf_log(message: str) -> None:
    print(f"[OPF] {message}", flush=True)


try:
    # fast nearest-neighbour
    from sklearn.neighbors import KDTree
    SK_HAS = True
except Exception:
    SK_HAS = False

try:
    # for optional clustering
    from sklearn.cluster import KMeans
    SK_CLUSTER = True
except Exception:
    SK_CLUSTER = False


# ---------- helpers
def _norm_country(x) -> str:
    return str(x or "").strip().lower()

def _haversine_km(lat1, lon1, lat2, lon2):
    # inputs in degrees, output in km
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _susceptance_from_x(x_ohm_per_unit: float, circuits: float = 1.0) -> float:
    # treat x as reactance in p.u. or per-line reactance proxy -> b = circuits / x
    # if x is zero-ish, clamp
    x = float(x_ohm_per_unit)
    if abs(x) < 1e-9:
        x = 1e-9
    return circuits / x

def _fmax_from_snom(s_nom_mva: float, circuits: float = 1.0) -> float:
    # DC model: MVA = MW
    return float(s_nom_mva) * float(circuits)


AC_EDGE_COLUMNS = ["ac_id", "bus0", "bus1", "voltage_kv", "x", "circuits", "s_nom", "b", "fmax"]
AC_EDGE_NUMERIC_COLUMNS = ["voltage_kv", "x", "circuits", "s_nom", "b", "fmax"]


def _safe_id_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("_") or "id"


def _empty_ac_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: pd.Series(dtype="float64" if col in AC_EDGE_NUMERIC_COLUMNS else "object")
            for col in AC_EDGE_COLUMNS
        }
    )


def _normalize_ac_edges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in AC_EDGE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out.loc[:, AC_EDGE_COLUMNS].copy()
    for col in AC_EDGE_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _concat_ac_edges(*frames: pd.DataFrame) -> pd.DataFrame:
    normalized = [_normalize_ac_edges(frame) for frame in frames if frame is not None and not frame.empty]
    if not normalized:
        return _empty_ac_edges()
    return pd.concat(normalized, axis=0, ignore_index=True)


# ---------- read raw CSVs
def read_network_csvs(
    *,
    buses_csv: Path,
    lines_csv: Path,
    transformers_csv: Path,
    links_csv: Path,
    converters_csv: Path | None = None,
    min_voltage_kv: int = 220
):
    """Read raw CSVs and return dataframes with consistent columns.

    Returns:
      df_buses: columns [bus_id, country, lat, lon, voltage_kv, is_dc]
      df_ac:    columns [ac_id, bus0, bus1, voltage_kv, x, circuits, s_nom, b, fmax]
      df_dc:    columns [dc_id, bus0, bus1, voltage_kv, p_nom, pmax]
    """
    # Buses
    B = pd.read_csv(buses_csv)
    # expected columns: bus_id, voltage, dc, lon, lat, country
    B = B.rename(columns=str.strip)
    B["country"] = B["country"].map(_norm_country)
    B["voltage"] = B["voltage"].astype(float)
    B = B[B["voltage"] >= float(min_voltage_kv)]
    B["is_dc"] = B.get("dc", False).astype(str).str.upper().isin(["TRUE", "1"])
    df_buses = B[["bus_id", "country", "lat", "lon", "voltage", "is_dc"]].rename(
        columns={"voltage": "voltage_kv"}
    ).copy()

    # AC lines
    L = pd.read_csv(lines_csv)
    L = L.rename(columns=str.strip)
    L["voltage"] = L["voltage"].astype(float)
    L = L[L["voltage"] >= float(min_voltage_kv)]
    L["circuits"] = L.get("circuits", 1.0).astype(float)
    L["s_nom"] = L.get("s_nom", 0.0).astype(float)
    L["x"] = L.get("x", 0.0).astype(float)
    L["b"] = L.apply(lambda r: _susceptance_from_x(r["x"], r["circuits"]), axis=1)
    L["fmax"] = L.apply(lambda r: _fmax_from_snom(r["s_nom"], r["circuits"]), axis=1)
    df_lines = L[["line_id", "bus0", "bus1", "voltage", "x", "circuits", "s_nom", "b", "fmax"]].rename(
        columns={"voltage": "voltage_kv", "line_id": "ac_id"}
    ).copy()

    # Transformers as AC edges (different voltages)
    T = pd.read_csv(transformers_csv)
    T = T.rename(columns=str.strip)
    # expected: transformer_id, bus0, bus1, voltage_bus0, voltage_bus1, s_nom
    T["circuits"] = 1.0
    T["x"] = T.get("x", 0.1)  # if no x provided, pick a mild reactance
    T["b"] = T.apply(lambda r: _susceptance_from_x(r["x"], 1.0), axis=1)
    T["fmax"] = T.apply(lambda r: _fmax_from_snom(r["s_nom"], 1.0), axis=1)
    df_trafos = T[["transformer_id", "bus0", "bus1", "x", "circuits", "s_nom", "b", "fmax"]].rename(
        columns={"transformer_id": "ac_id"}
    ).copy()
    df_trafos["voltage_kv"] = np.nan  # mixed

    # unify AC edges
    df_ac = _concat_ac_edges(df_lines, df_trafos)

    # DC links
    D = pd.read_csv(links_csv)
    D = D.rename(columns=str.strip)
    D["p_nom"] = D.get("p_nom", 0.0).astype(float)
    D["pmax"] = D["p_nom"].abs()
    df_dc = D[["link_id", "bus0", "bus1", "voltage", "p_nom", "pmax"]].rename(
        columns={"voltage": "voltage_kv", "link_id": "dc_id"}
    ).copy()

    # Some converter files contain bus ids already in links; we only need links here.
    return df_buses, df_ac, df_dc


# ---------- clustering & contraction (AC only)
def cluster_contract(
    *,
    df_buses: pd.DataFrame,
    df_ac: pd.DataFrame,
    df_dc: pd.DataFrame,
    k_per_country: Dict[str, int] | None = None,  # e.g. {"de": 80, "fr": 50, ...}
    k_default: int = 30
):
    """Contract an AC grid by clustering buses within each country.

    DC links are not clustered; only their endpoints are mapped to the reduced
    AC buses. The returned AC corridors aggregate all equivalent edges between
    the same reduced endpoints.
    """
    B = df_buses.copy()
    AC = df_ac.copy()
    DC = df_dc.copy()

    # build per-country clustering targets
    country_groups = B.groupby("country")
    mapping = {}
    nodes = []
    seq = 0

    for c, G in country_groups:
        n = len(G)
        if n == 0:
            continue
        K = int((k_per_country or {}).get(c, min(n, k_default)))
        if K <= 1:
            # no reduction
            for _, r in G.iterrows():
                nid = r["bus_id"]
                mapping[r["bus_id"]] = nid
                nodes.append((nid, c, float(r["lat"]), float(r["lon"])))
        else:
            if not SK_CLUSTER:
                # simple: pick K random representatives (or KMeans substitute)
                idx = np.linspace(0, n - 1, K).astype(int)
                reps = G.iloc[idx]
                for _, r in G.iterrows():
                    # nearest representative by haversine
                    d = _haversine_km(r["lat"], r["lon"], reps["lat"].values, reps["lon"].values)
                    j = int(np.argmin(d))
                    nid = f"{c.upper()}_CL{seq}_{j}"
                    mapping[r["bus_id"]] = nid
                # centers = rep coords
                for j, (_, rr) in enumerate(reps.iterrows()):
                    nid = f"{c.upper()}_CL{seq}_{j}"
                    nodes.append((nid, c, float(rr["lat"]), float(rr["lon"])))
            else:
                XY = np.c_[G["lat"].values, G["lon"].values]
                kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
                lab = kmeans.fit_predict(XY)
                centers = kmeans.cluster_centers_
                for (bus_id, lab_i) in zip(G["bus_id"].values, lab):
                    nid = f"{c.upper()}_CL{seq}_{int(lab_i)}"
                    mapping[bus_id] = nid
                for j, (lat, lon) in enumerate(centers):
                    nid = f"{c.upper()}_CL{seq}_{j}"
                    nodes.append((nid, c, float(lat), float(lon)))
            seq += 1

    buses_red = pd.DataFrame(nodes, columns=["bus_id", "country", "lat", "lon"]).drop_duplicates("bus_id")

    # contract AC lines to corridors
    def _endpoints(u, v):
        a, b = mapping.get(u, u), mapping.get(v, v)
        return (a, b) if a <= b else (b, a)

    agg = defaultdict(lambda: {"b_sum": 0.0, "fmax_sum": 0.0, "n_parallel": 0})
    for _, r in AC.iterrows():
        a, b = _endpoints(r["bus0"], r["bus1"])
        if a == b:
            continue  # internal to a cluster
        agg[(a, b)]["b_sum"] += float(r["b"])
        agg[(a, b)]["fmax_sum"] += float(r["fmax"])
        agg[(a, b)]["n_parallel"] += 1

    rows = []
    for (a, b), val in agg.items():
        rows.append({
            "corr_id": f"ac_{a}__{b}",
            "n_from": a, "n_to": b,
            "b_sum": val["b_sum"], "fmax_sum": val["fmax_sum"], "n_parallel": int(val["n_parallel"])
        })
    ac_corr = pd.DataFrame(rows)

    # map DC endpoints
    dc_rows = []
    for _, r in DC.iterrows():
        a, b = mapping.get(r["bus0"], r["bus0"]), mapping.get(r["bus1"], r["bus1"])
        if a == b:
            continue
        dc_rows.append({"dc_id": r["dc_id"], "n_from": a, "n_to": b, "pmax": float(r["pmax"])})
    dc_links = pd.DataFrame(dc_rows)

    return buses_red, ac_corr, dc_links, mapping


# ---------- assign plants to buses
def assign_plants_to_buses(
    *,
    plants_latlon: Dict[str, Tuple[float, float]],
    plants_country: Dict[str, str],
    buses_red: pd.DataFrame
) -> Dict[str, str]:
    """Map each plant to the nearest reduced bus in the same country.

    A global nearest-bus fallback is used only when no reduced bus exists for the
    reported plant country. This keeps the mapping complete while making such
    cases visible through the resulting country/bus assignments.
    """
    by_country = {}
    for c, G in buses_red.groupby("country"):
        if len(G) == 0:
            continue
        pts = np.c_[G["lat"].values, G["lon"].values]
        if SK_HAS and len(G) > 1:
            kd = KDTree(np.radians(pts))  # radians for haversine
            by_country[c] = (G["bus_id"].values, kd, pts)
        else:
            by_country[c] = (G["bus_id"].values, None, pts)

    mapping = {}
    for p, (lat, lon) in plants_latlon.items():
        c = _norm_country(plants_country[p])
        if c in by_country and len(by_country[c][0]) > 0:
            ids, kd, pts = by_country[c]
            if kd is not None:
                dist, ind = kd.query(np.radians([[lat, lon]]), k=1)
                j = int(ind[0, 0])
            else:
                d = _haversine_km(lat, lon, pts[:, 0], pts[:, 1])
                j = int(np.argmin(d))
            mapping[p] = str(ids[j])
        else:
            # fallback global nearest
            G = buses_red
            pts = np.c_[G["lat"].values, G["lon"].values]
            d = _haversine_km(lat, lon, pts[:, 0], pts[:, 1])
            j = int(np.argmin(d))
            mapping[p] = str(G["bus_id"].values[j])
    return mapping


# ---------- distribute country series to buses
def distribute_series_to_buses(
    *,
    series_ycw: Dict[Tuple[int, str, int], float],  # {(y, country, w): value}
    buses_red: pd.DataFrame,
    bus_weights: Dict[str, float] | None = None
) -> Dict[Tuple[int, str, int], float]:
    """Distribute country-week values to buses using optional bus weights."""

    idx_by_country = defaultdict(list)
    for i, r in buses_red[["bus_id", "country"]].iterrows():
        idx_by_country[r["country"]].append(r["bus_id"])

    # normalize weights per country
    if bus_weights is None:
        bus_weights = {}
    weights_by_country = {}
    for c, bus_list in idx_by_country.items():
        ww = np.array([float(bus_weights.get(n, 1.0)) for n in bus_list], dtype=float)
        if ww.sum() <= 0:
            ww = np.ones(len(bus_list), dtype=float)
        ww = ww / ww.sum()
        weights_by_country[c] = dict(zip(bus_list, ww.tolist()))

    out = {}
    for (y, c, w), val in series_ycw.items():
        bus_list = idx_by_country.get(c, [])
        if not bus_list:
            continue
        wb = weights_by_country[c]
        for n in bus_list:
            out[(y, n, w)] = float(val) * float(wb.get(n, 0.0))
    return out


def _norm_country_upper(x: Any) -> str:
    c = str(x or "").strip().upper()
    if c == "UK":
        return "GB"
    if c == "EL":
        return "GR"
    return c


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _first_existing(frame: pd.DataFrame, candidates: list[str], default: Any = None) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series([default] * len(frame), index=frame.index)


def _resolve_converter_map(converters_csv: Path | None) -> dict[str, str]:
    if converters_csv is None or not Path(converters_csv).exists():
        return {}

    converters = pd.read_csv(converters_csv, sep=";", low_memory=False).rename(columns=str.strip)
    req = {"bus0", "bus1"}
    if not req.issubset(converters.columns):
        return {}

    mapping: dict[str, str] = {}
    for row in converters.itertuples(index=False):
        bus0 = str(getattr(row, "bus0", "")).strip()
        bus1 = str(getattr(row, "bus1", "")).strip()
        if bus0.startswith("cl_dc_keep_way/") and bus1:
            mapping[bus0] = bus1
        elif bus1.startswith("cl_dc_keep_way/") and bus0:
            mapping[bus1] = bus0
    return mapping


def read_reduced_network_csvs(
    *,
    buses_csv: Path,
    lines_csv: Path,
    transformers_csv: Path,
    links_csv: Path,
    converters_csv: Path | None = None,
    min_voltage_kv: int = 220
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and normalize the reduced network dataset used by the OPF scripts.

    The input data may contain converters, transformers, AC lines, and DC links
    in slightly different table conventions. This function returns a consistent
    bus/AC/DC representation and removes edges whose endpoints are not part of
    the selected high-voltage bus set.
    """
    _opf_log(
        "Reading reduced network CSVs: "
        f"buses={buses_csv}, lines={lines_csv}, transformers={transformers_csv}, links={links_csv}"
    )

    buses = pd.read_csv(buses_csv, sep=";", low_memory=False).rename(columns=str.strip)
    buses["country"] = buses["country"].map(_norm_country_upper)
    buses["voltage"] = pd.to_numeric(buses["voltage"], errors="coerce")
    buses["lat"] = pd.to_numeric(buses["lat"], errors="coerce")
    buses["lon"] = pd.to_numeric(buses["lon"], errors="coerce")
    buses["is_dc"] = _first_existing(buses, ["dc"], False).map(_as_bool)
    buses = buses.dropna(subset=["bus_id", "country", "voltage"]).copy()
    buses = buses[buses["voltage"] >= float(min_voltage_kv)].copy()

    df_buses = buses[["bus_id", "country", "lat", "lon", "voltage", "is_dc"]].rename(
        columns={"voltage": "voltage_kv"}
    )

    lines = pd.read_csv(lines_csv, sep=";", low_memory=False).rename(columns=str.strip)
    if not lines.empty:
        lines["u"] = lines["u"].astype(str)
        lines["v"] = lines["v"].astype(str)
        lines["voltage"] = pd.to_numeric(lines["voltage"], errors="coerce")
        lines["s_nom"] = pd.to_numeric(lines["s_nom"], errors="coerce").fillna(0.0)
        lines["circuits"] = pd.to_numeric(_first_existing(lines, ["circuits"], 1.0), errors="coerce").fillna(1.0)
        lines["x_eq"] = pd.to_numeric(_first_existing(lines, ["x_eq", "x"], 0.0), errors="coerce").fillna(0.0)
        lines["b"] = lines.apply(
            lambda r: _as_float(_first_existing(lines.loc[[r.name]], ["b"], np.nan).iloc[0], np.nan)
            if "b" in lines.columns
            else _susceptance_from_x(r["x_eq"], r["circuits"]),
            axis=1,
        )
        lines["fmax"] = lines.apply(lambda r: _fmax_from_snom(r["s_nom"], r["circuits"]), axis=1)
        lines = lines[lines["voltage"] >= float(min_voltage_kv)].copy()
        df_lines = lines[["line_id", "u", "v", "voltage", "x_eq", "circuits", "s_nom", "b", "fmax"]].rename(
            columns={"line_id": "ac_id", "u": "bus0", "v": "bus1", "voltage": "voltage_kv", "x_eq": "x"}
        )
    else:
        df_lines = pd.DataFrame(columns=["ac_id", "bus0", "bus1", "voltage_kv", "x", "circuits", "s_nom", "b", "fmax"])

    trafos = pd.read_csv(transformers_csv, sep=";", low_memory=False).rename(columns=str.strip)
    if not trafos.empty:
        trafos["bus0_red"] = trafos["bus0_red"].astype(str)
        trafos["bus1_red"] = trafos["bus1_red"].astype(str)
        trafos["s_nom"] = pd.to_numeric(trafos["s_nom"], errors="coerce").fillna(0.0)
        trafos["circuits"] = pd.to_numeric(_first_existing(trafos, ["n_tr"], 1.0), errors="coerce").fillna(1.0)
        trafos["x"] = pd.to_numeric(_first_existing(trafos, ["x_eq", "x"], 0.1), errors="coerce").fillna(0.1)
        trafos["b"] = trafos.apply(lambda r: _susceptance_from_x(r["x"], r["circuits"]), axis=1)
        trafos["fmax"] = trafos.apply(lambda r: _fmax_from_snom(r["s_nom"], r["circuits"]), axis=1)
        df_trafos = trafos[["transformer_id", "bus0_red", "bus1_red", "x", "circuits", "s_nom", "b", "fmax"]].rename(
            columns={"transformer_id": "ac_id", "bus0_red": "bus0", "bus1_red": "bus1"}
        )
        df_trafos["voltage_kv"] = np.nan
    else:
        df_trafos = pd.DataFrame(columns=["ac_id", "bus0", "bus1", "x", "circuits", "s_nom", "b", "fmax", "voltage_kv"])

    df_ac = _concat_ac_edges(df_lines, df_trafos)

    links = pd.read_csv(links_csv, sep=";", low_memory=False).rename(columns=str.strip)
    converter_map = _resolve_converter_map(converters_csv)
    if not links.empty:
        links["bus0"] = links["bus0"].astype(str).map(lambda x: converter_map.get(x, x))
        links["bus1"] = links["bus1"].astype(str).map(lambda x: converter_map.get(x, x))
        links["voltage"] = pd.to_numeric(links["voltage"], errors="coerce").fillna(0.0)
        links["p_nom"] = pd.to_numeric(_first_existing(links, ["p_nom"], 0.0), errors="coerce").fillna(0.0)
        links["n_links"] = pd.to_numeric(_first_existing(links, ["n_links", "n_link", "circuits"], 1.0), errors="coerce").fillna(1.0)
        links["pmax"] = links["p_nom"].abs()
        df_dc = links[["link_id", "bus0", "bus1", "voltage", "p_nom", "pmax", "n_links"]].rename(
            columns={"link_id": "dc_id", "voltage": "voltage_kv"}
        )
    else:
        df_dc = pd.DataFrame(columns=["dc_id", "bus0", "bus1", "voltage_kv", "p_nom", "pmax", "n_links"])

    valid_buses = set(df_buses["bus_id"].astype(str))
    df_ac = df_ac[df_ac["bus0"].astype(str).isin(valid_buses) & df_ac["bus1"].astype(str).isin(valid_buses)].copy()
    df_dc = df_dc[df_dc["bus0"].astype(str).isin(valid_buses) & df_dc["bus1"].astype(str).isin(valid_buses)].copy()

    _opf_log(
        "Reduced network CSVs loaded: "
        f"buses={len(df_buses)}, ac_edges={len(df_ac)}, dc_links={len(df_dc)}"
    )
    return df_buses.reset_index(drop=True), df_ac.reset_index(drop=True), df_dc.reset_index(drop=True)


def build_reduced_network_topology(
    *,
    df_buses: pd.DataFrame,
    df_ac: pd.DataFrame,
    df_dc: pd.DataFrame,
    disaggregate_parallel_ac: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build the AC/DC topology tables consumed by the optimization model.

    By default AC edges are aggregated to endpoint corridors. With
    disaggregate_parallel_ac=True, each reduced AC edge is expanded into
    single-circuit model elements while parent_corr_id keeps the aggregated
    corridor identity for outputs and visualisation.
    """
    _opf_log(
        "Building reduced network topology: "
        f"raw_buses={len(df_buses)}, raw_ac_edges={len(df_ac)}, raw_dc_links={len(df_dc)}"
    )

    buses_red = df_buses.copy()
    buses_red["bus_id"] = buses_red["bus_id"].astype(str)
    buses_red["country"] = buses_red["country"].map(_norm_country_upper)

    def _ordered_endpoints(u: Any, v: Any) -> tuple[str, str]:
        a = str(u)
        b = str(v)
        return (a, b) if a <= b else (b, a)

    line_to_corr: dict[str, str] = {}
    ac_rows: list[dict[str, Any]] = []

    if disaggregate_parallel_ac:
        circuit_seq = 0
        for row in df_ac.itertuples(index=False):
            n0, n1 = _ordered_endpoints(row.bus0, row.bus1)
            if n0 == n1:
                continue
            parent_corr_id = f"ac_{n0}__{n1}"
            source_id = str(getattr(row, "ac_id", f"ac_edge_{circuit_seq}"))
            n_units = max(1, int(round(_as_float(getattr(row, "circuits", 1.0), 1.0))))
            b_total = _as_float(getattr(row, "b", 0.0))
            fmax_total = _as_float(getattr(row, "fmax", 0.0))
            line_to_corr[source_id] = parent_corr_id
            for unit_idx in range(1, n_units + 1):
                circuit_seq += 1
                ac_rows.append(
                    {
                        "corr_id": f"{parent_corr_id}__{_safe_id_token(source_id)}__c{unit_idx:02d}",
                        "parent_corr_id": parent_corr_id,
                        "source_ac_id": source_id,
                        "source_circuits": n_units,
                        "n_from": n0,
                        "n_to": n1,
                        "b_sum": b_total / float(n_units),
                        "fmax_sum": fmax_total / float(n_units),
                        "n_parallel": 1,
                    }
                )
    else:
        ac_agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"b_sum": 0.0, "fmax_sum": 0.0, "n_parallel": 0.0}
        )
        for row in df_ac.itertuples(index=False):
            n0, n1 = _ordered_endpoints(row.bus0, row.bus1)
            if n0 == n1:
                continue
            corr_id = f"ac_{n0}__{n1}"
            ac_agg[(n0, n1)]["b_sum"] += _as_float(getattr(row, "b", 0.0))
            ac_agg[(n0, n1)]["fmax_sum"] += _as_float(getattr(row, "fmax", 0.0))
            ac_agg[(n0, n1)]["n_parallel"] += max(1.0, _as_float(getattr(row, "circuits", 1.0), 1.0))
            line_to_corr[str(row.ac_id)] = corr_id

        ac_rows = [
            {
                "corr_id": f"ac_{a}__{b}",
                "parent_corr_id": f"ac_{a}__{b}",
                "source_ac_id": "",
                "source_circuits": int(round(vals["n_parallel"])),
                "n_from": a,
                "n_to": b,
                "b_sum": vals["b_sum"],
                "fmax_sum": vals["fmax_sum"],
                "n_parallel": int(round(vals["n_parallel"])),
            }
            for (a, b), vals in sorted(ac_agg.items())
        ]
    ac_corr = pd.DataFrame(ac_rows)

    dc_rows = []
    for row in df_dc.itertuples(index=False):
        if str(row.bus0) == str(row.bus1):
            continue
        dc_rows.append(
            {
                "dc_id": str(row.dc_id),
                "n_from": str(row.bus0),
                "n_to": str(row.bus1),
                "pmax": _as_float(getattr(row, "pmax", 0.0)),
                "n_parallel": max(1, int(round(_as_float(getattr(row, "n_links", 1.0), 1.0)))),
            }
        )
    dc_links = pd.DataFrame(dc_rows)
    if not dc_links.empty:
        dc_links = dc_links.drop_duplicates(subset=["dc_id"]).reset_index(drop=True)

    _opf_log(
        "Reduced network topology built: "
        f"buses={len(buses_red)}, ac_corridors={len(ac_corr)}, dc_links={len(dc_links)}, "
        f"disaggregate_parallel_ac={bool(disaggregate_parallel_ac)}"
    )
    return buses_red.reset_index(drop=True), ac_corr.reset_index(drop=True), dc_links, line_to_corr
