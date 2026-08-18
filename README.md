# Integrated Generator and Transmission Maintenance Scheduling

This repository contains the research code used to construct, optimize, and
evaluate weekly generator maintenance schedules (GMS) and transmission
maintenance schedules (TMS) for reduced European power-system cases. The paper
cases cover Actual 2025 and TYNDP 2024 target years 2030 and 2040.

The code is a research model, not an operational outage-coordination tool.

## Model Scope

Maintenance decisions are shared across all selected weather years. Dispatch
and network operation are scenario-dependent recourse decisions evaluated for
one sampled peak-load hour in every model week.

The implementation supports:

- grouped thermal GMS,
- AC-corridor and DC-link TMS in OPF cases,
- stochastic weather-year scenarios and probability weights,
- reduced-grid DC optimal power flow,
- national economic dispatch on country nodes,
- a constructive GMS/TMS heuristic and fixed-schedule evaluation,
- monolithic MIP optimization and optional Benders decomposition, and
- publication tables and figures for the paper cases.

The operating model is linear. It does not include unit commitment, startup
costs, ramping, minimum up/down times, or dynamic stability constraints.
Frequency-reserve requirements are added to country load and may be supplied at
any bus in the same country. A reserve shortage is therefore represented by
energy not served (ENS); there are no technology-specific reserve variables or
a separate FR-slack objective.

## Mathematical Structure

### First stage

The common first-stage plan contains thermal maintenance starts and weekly
availability, optional long-maintenance assignments, and, in OPF mode,
AC-corridor and DC-link maintenance. The constraints enforce maintenance
duration and frequency, horizon boundaries, country-week concurrency, CHP
winter exclusions, configured long-maintenance shares, corridor/link
concurrency, and border maintenance-capacity limits.

### Second stage

For each weather-year/week pair, the recourse model determines thermal and
other controllable generation, renewable generation and curtailment, hydro and
battery operation where represented, demand-side response, ENS, network flows,
and energy balances.

### Network modes

- `NETWORK_MODE="opf"` uses nodal balances and DC network constraints. The
  default `FLOW_FORMULATION="theta"` enforces angle-based DC power flow; `ptdf`
  is available as an explicit alternative. TMS is available only in this mode.
- `NETWORK_MODE="ed_national"` aggregates resources and balances to one node
  per country and uses transport limits between countries. Kirchhoff and Ohm
  constraints are omitted, and individual transmission assets are not
  scheduled. Transfer limits are selected with
  `NATIONAL_ED_CAPACITY_SOURCE="line_aggregate"` or `"ntc"`.

The country export-shortage guard prevents a country from being a net exporter
while it has ENS. Transit remains possible because the restriction applies to
net exports rather than gross cross-border flows.

## Objectives

The code default is a single expected-ENS objective:

```python
OBJECTIVE_ORDER = ("ens",)
PRIMARY_OBJECTIVE = "ens"
```

Expected ENS is weighted with the configured weather-year probabilities.
Additional objective keys are retained for explicit experiments:

- `ens_self_supply`: ENS plus soft national self-supply slack,
- `self_supply_slack`: normalized soft national self-supply violations,
- `self_supply_slack_power`: soft national self-supply violations in power
  units,
- `europe_reliability_index`: Europe-wide relative capacity reserve,
- `europe_reliability_ens`: reliability index with an ENS penalty,
- `line_capacity_margin`: minimum weekly available transfer-capacity share,
- `inertia_availability`: minimum weekly available thermal inertia potential.

A multi-entry `OBJECTIVE_ORDER` is solved lexicographically. The paper workflow
uses the single-stage `ens` objective.

## Code Defaults

The following values are the actual defaults in
`model/optimization_tyndp_opf.py`, not a recommended override profile:

```python
NETWORK_MODE = "opf"
FLOW_FORMULATION = "theta"
COUNTRY_EXPORT_SHORTAGE_GUARD = True
WARM_START_HEURISTIC = False
FIX_LINE_MAINTENANCE_FROM_HEURISTIC = False
```

Consequently, a direct unmodified optimization is an angle-based nodal OPF
with the export-shortage guard. It neither loads a heuristic MIP start nor fixes
the heuristic TMS. The default solution method is the monolithic MIP
(`HEURISTIC=False`, `BENDERS=False`) with `OBJECTIVE_ORDER=("ens",)`.

The matrix runners deliberately override selected defaults for a requested
experiment. Every run writes the effective configuration to `run_config.json`.

## Paper Workflow

The primary paper workflow separates TMS construction from GMS optimization:

1. Run the schedule-only heuristic once for each case. It constructs both a
   thermal GMS and an AC/DC TMS.
2. Evaluate the fixed heuristic schedule to obtain a directly comparable
   heuristic ENS result.
3. Copy the heuristic schedule files to the scenario-specific warm-start
   directory.
4. Run the monolithic MIP with the heuristic TMS fixed and optimize the GMS.

For step 4, the decisive setting is:

```python
FIX_LINE_MAINTENANCE_FROM_HEURISTIC = True
```

The thermal GMS remains a decision of the MIP. In a matrix-runner
`warm-optimizations` stage, `WARM_START_HEURISTIC=True` also supplies the
heuristic GMS as an initial incumbent, but it does not fix the GMS. To reproduce
the same fixed-TMS method without a thermal MIP start in a direct run, keep
`WARM_START_HEURISTIC=False` and set only
`FIX_LINE_MAINTENANCE_FROM_HEURISTIC=True`.

This workflow reduces the integer search space while preserving optimization
of generator outages. It should be reported as fixed heuristic TMS plus
optimized GMS, rather than as joint GMS/TMS optimization.

## Maintenance-Year Profiles

Both profiles contain 52 model weeks; only their mapping to calendar weeks and
weather data differs:

- `jan_dec` (Jan-Dec): model week 1 corresponds to calendar week 1, and the
  horizon covers calendar weeks 1-52 of the same year.
- `w17_w16` (Apr-Apr): model week 1 corresponds to ISO calendar week 17
  (approximately April), followed by weeks 17-52 and weeks 1-16 of the next
  year. This profile is labelled Apr-Apr in publication figures.

The shifted profile requires a source-week mapping because one maintenance
year combines two adjacent calendar years. Maintenance constraints still use
model-week indices 0-51 internally. The exported
`opf_maintenance_year_week_mapping.csv` records the calendar mapping.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `model/optimization_tyndp_opf.py` | Configuration and entry point for one case. |
| `model/preprocess_tyndp_opf.py` | Input harmonization and solver data construction. |
| `model/solve_tyndp_opf.py` | Monolithic MIP, Benders, fixed-schedule evaluation, and exports. |
| `model/solve_tyndp_opf_heuristic.py` | Constructive GMS/TMS heuristic and feasibility recourse. |
| `model/run_tyndp2024_matrix.py` | TYNDP 2030/2040 experiment matrix. |
| `model/run_actual_2025_matrix.py` | Actual-2025 experiment matrix. |
| `model/matrix_common.py` | Shared matrix-runner configuration and naming helpers. |
| `visualisation/build_modular_publication_figures.py` | Main publication figures. |
| `RUN_WEATHER_YEAR_EXPERIMENTS.md` | Complete portable experiment runbook. |

## Environment

Create the maintained Conda environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate maint-model
```

For a solver-validated Windows environment, use
`environment.win-64.lock.yml`. A valid Gurobi installation and license are
required.

## Input Data

The datasets are intentionally stored outside the repository. Set an absolute
input root and pass it to the matrix runners:

```bash
export INPUT_ROOT="/absolute/path/to/revision_outage_input"
export OUTPUT_ROOT="/absolute/path/to/revision_outage_output"
```

The input discovery code expects the following structure. `<year>` is 2025,
2030, or 2040 and `<network>` is the full reduced-network directory name.
Files below a domain/year directory may be nested further when the discovery
code searches recursively.

```text
<INPUT_ROOT>/
|-- grid/target_year_<year>/<network>/
|   |-- buses.csv
|   |-- plants.csv
|   |-- lines.csv
|   |-- transformers.csv
|   |-- links.csv
|   |-- converters.csv
|   `-- buses_with_clusters.csv
|-- load/target_year_<year>/<network>/
|   `-- disaggregated_load_country_bus_load_pop40_gdp60.csv
|-- powerplants/target_year_<year>/<network>/
|   |-- thermal/thermal_units.csv
|   |-- other_res/other_res_capacity_country_bus.csv
|   |-- other_res/other_res_availability_country_bus_weekly.csv
|   |-- other_nonres/other_nonres_capacity_country_bus.csv
|   `-- other_nonres/other_nonres_availability_country_bus_weekly.csv
|-- renewables/target_year_<year>/<network>/.../
|   `-- disaggregated_res_country_bus.csv
|-- hydro/target_year_<year>/<network>/
|   |-- disaggregated_hydro_bus_capacities.csv
|   `-- disaggregated_hydro_bus_constraints_weekly.csv
|-- bess/target_year_<year>/<network>/bess_capacity_country_bus.csv
|-- dsr/target_year_<year>/<network>/
|   |-- dsr_capacity_country_bus.csv
|   `-- dsr_availability_country_bus_weekly.csv
|-- transmission/target_year_<year>/<network>/
|   |-- ntc_tyndp2024.csv
|   `-- country_aggregation_map_<year>_tyndp2024.csv
|-- weather_year_reduction/
|   |-- target_year_<year>/...
|   `-- scenarios/w17_w16/target_year_<year>/...
|-- warm_start/                         # generated by the matrix workflow
|-- frequency_reserves_<year>_tyndp2024.csv
|-- inertia_factors_entsoe.csv
|-- plants_max_weekly_revisions_country.csv
|-- plants_median_revision_duration_weeks_country_2015-2025_planned.csv
|-- plants_max_revision_duration_weeks_country_2015-2025_planned.csv
|-- plants_std_revision_duration_weeks_country_<year>_tyndp2024.csv
`-- plants_long_revision_duration_weeks_country_<year>_tyndp2024.csv
```

Network and load files, thermal units, renewable availability, reserve demand,
weather weights, and maintenance constraints are required for the respective
case. Hydro, storage, DSR, and other-generation files are required when those
resources are enabled. `ntc_tyndp2024.csv` is required for national ED with
`NATIONAL_ED_CAPACITY_SOURCE="ntc"`; the reduced-grid asset files are required
for OPF/TMS. Inertia factors are needed only for inertia outputs or objectives.

For `w17_w16`, each target year additionally requires
`source_weather_week_schedule.csv`, a weather-year selection file for reduced
sets such as `k07`, and the corresponding probability file. Exact paths are
listed in the runbook.

## Reproducing Paper Runs

The complete commands, including both maintenance-year profiles, export-guard
variants, heuristic evaluation, warm-start preparation, and resume behavior,
are in [`RUN_WEATHER_YEAR_EXPERIMENTS.md`](RUN_WEATHER_YEAR_EXPERIMENTS.md).

The fixed-TMS paper pipeline is selected by adding
`--fix-line-maintenance-from-heuristic` to the warm optimization stage. A
minimal TYNDP sequence is:

```bash
python model/run_tyndp2024_matrix.py \
  --stage pure-heuristics --years 2030 2040 --weather k07 \
  --models k128 k256 --network-modes opf ed_national \
  --maintenance-year-profiles jan_dec w17_w16 \
  --dir-base "$INPUT_ROOT" --dir-out "$OUTPUT_ROOT/opf_tyndp2024"

python model/run_tyndp2024_matrix.py \
  --stage prepare-warm-start --years 2030 2040 --weather k07 \
  --models k128 k256 --network-modes opf ed_national \
  --maintenance-year-profiles jan_dec w17_w16 \
  --dir-base "$INPUT_ROOT" --dir-out "$OUTPUT_ROOT/opf_tyndp2024"

python model/run_tyndp2024_matrix.py \
  --stage warm-optimizations --years 2030 2040 --weather k07 \
  --models k128 k256 --network-modes opf ed_national \
  --maintenance-year-profiles jan_dec w17_w16 \
  --objective-preset ens --fix-line-maintenance-from-heuristic \
  --dir-base "$INPUT_ROOT" --dir-out "$OUTPUT_ROOT/opf_tyndp2024"
```

Use the runbook rather than the abbreviated sequence for publication results;
it keeps batch IDs, warm-start namespaces, and export-guard variants aligned.

The monolithic MIP is the default. Benders must be selected explicitly:

- TYNDP runner: `--benders`
- Actual-2025 runner: `--workflow benders` (or its `--benders` alias)
- direct single-case configuration: `BENDERS=True`

## Reproducing Publication Figures

Point the figure builder to the common output root and generate PDF artifacts:

```bash
export REVISION_OUTAGE_OUTPUT="$OUTPUT_ROOT"

python visualisation/build_modular_publication_figures.py \
  --dataset all --profiles jan_dec w17_w16 --formats pdf
```

The builder reads the timestamped run directories below
`$OUTPUT_ROOT/opf_actual_2025` and `$OUTPUT_ROOT/opf_tyndp2024`. Dedicated base
and historical-comparison figure scripts have their own `--help` output because
they require additional source datasets that are not distributed with this
repository.

## Outputs and Reproducibility

Each run writes a timestamped directory with configuration metadata,
maintenance schedules, solver status and timings, ENS and dispatch results,
network flows, and workflow diagnostics. Publication comparisons should record
at least the git commit, input-data version, weather years and probabilities,
maintenance-year profile, network mode and aggregation, objective order,
export-guard setting, whether TMS was fixed or optimized, warm-start source,
and Gurobi version and parameters.

Raw power data are generally in MW. With `SCALE_POWER_TO_GW=True`, the solver
uses GW for numerical conditioning. A weekly ENS value represents one sampled
one-hour state, so energy equals power multiplied by one hour; no factor of 168
is applied.

## Verification

```bash
python -m pytest -q
ruff check model preprocessing visualisation tests
```
