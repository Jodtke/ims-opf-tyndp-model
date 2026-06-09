# Stochastic Integrated Generator and Transmission Maintenance Scheduling

This directory contains the optimization and evaluation workflow used for
integrated weekly generator and transmission maintenance scheduling in reduced
European power-system scenarios. The model combines thermal generator
maintenance scheduling, AC/DC transmission outage scheduling, stochastic
weather-year dispatch evaluation, frequency-reserve requirements, and
network-constrained recourse decisions.

The code is written for research use. It is meant to support reproducible
experiments for a publication workflow, not to replace an operational outage
coordination platform.

## Scope

The model addresses a mid-term maintenance planning problem with weekly
resolution. Maintenance decisions are made for one target year and are shared
across all considered weather years. Dispatch, reserve provision, curtailment,
and network flows are evaluated separately for each weather-year and week.

The current publication workflow supports:

- thermal generator maintenance scheduling,
- AC corridor and DC link maintenance scheduling,
- stochastic weather-year weighting and reduction,
- weekly OPF-based recourse evaluation,
- frequency-reserve adequacy checks,
- capacity-margin based objective terms,
- a constructive heuristic for generator and line maintenance,
- a compact MIP formulation for smaller cases and fixed-schedule evaluations,
- a Benders decomposition in which weekly scenario OPF blocks are separated
  from the master maintenance problem,
- warm starts from heuristic generator schedules, and
- fixed transmission maintenance schedules imported from the heuristic.

The genetic algorithm and augmented multi-objective workflows are intentionally
not documented here because they are not part of the current publication setup.

## Repository Layout

The main workflow is split across five scripts:

| File | Role |
| --- | --- |
| `optimization_tyndp_opf.py` | Main run script. Defines the target year, input and output paths, scenario selection, model switches, objective weights, solver parameters, and selected solution workflow. |
| `preprocess_tyndp_opf.py` | Reads and harmonizes input data for one target year. Builds country, node, technology, scenario, demand, generation, reserve, and maintenance parameter structures used by the solver. |
| `network_build.py` | Builds the reduced network representation from buses, AC branches, transformers, DC links, converters, and cluster metadata. Provides the topology used by the OPF formulation. |
| `solve_tyndp_opf.py` | Contains the mathematical optimization model, OPF recourse formulation, Benders decomposition, fixed-schedule evaluation, and output writing routines. |
| `solve_tyndp_opf_heuristic.py` | Implements the constructive maintenance heuristic, OPF-based schedule evaluation, and local recourse repair logic for difficult transmission schedules. |

Additional helper scripts may be present for diagnostics or post-processing.
They are not required for the standard model run.

## Conceptual Workflow

The model workflow is:

1. Select a target year, weather years, input data roots, and solver mode in
   `optimization_tyndp_opf.py`.
2. Read and validate the reduced grid and technology data.
3. Aggregate and map input data to the optimization sets: countries, nodes,
   thermal groups, AC corridors, DC links, storage units, renewable units, and
   weather-year scenarios.
4. Construct maintenance parameters such as outage durations, weekly limits,
   thermal group sizes, long-revision eligibility, and line-maintenance
   frequencies.
5. Solve the selected workflow:
   - heuristic scheduling and OPF evaluation,
   - compact MIP,
   - Benders decomposition, or
   - Benders with fixed transmission maintenance and optional generator warm
     start.
6. Write maintenance schedules, dispatch results, adequacy indicators, flow
   results, solver logs, and run configuration metadata to the output directory.

The publication workflow currently uses the heuristic to generate a feasible
transmission maintenance schedule. This schedule can then be fixed in the
optimization model. The reason is computational: the exact mixed-integer
representation of line outage states and AC susceptance changes introduces
large big-M formulations and binary topology variables. For the considered
European-scale stochastic problem, these variables made the integrated
transmission maintenance model difficult to solve and prevented robust
convergence even after scenario reduction. Fixing the TMS schedule preserves a
network-aware transmission outage plan while keeping the subsequent generator
maintenance optimization computationally tractable.

## Input Data

The scripts expect a prepared TYNDP-style data directory. The exact paths are
configured in `optimization_tyndp_opf.py`, especially through `DIR_BASE` and the
`FILES` dictionary. The preprocessing code uses structured file discovery
rather than relying on one single monolithic input file.

Typical input domains are:

- reduced network topology,
- country and node mappings,
- thermal power-plant data,
- thermal maintenance durations and limits,
- demand time series,
- renewable availability and capacity data,
- hydro storage and reservoir availability,
- battery storage data,
- other non-renewable controllable capacity,
- frequency-reserve requirements,
- transmission maintenance assumptions,
- optional net transfer capacity data,
- representative weather-year selection and weights.

The reduced grid usually contains:

- `buses.csv`,
- `lines.csv`,
- `transformers.csv`,
- `links.csv`,
- `converters.csv`, and
- `buses_with_clusters.csv`.

The model is designed for a reduced European system representation. If the
input data are replaced, the country labels, node labels, technology naming,
line identifiers, and weather-year metadata must remain consistent across all
input domains.

## Units and Scaling

Most raw power-system inputs are given in MW. The run script can scale power
quantities internally, for example from MW to GW, through `SCALE_POWER_TO_GW`.
This improves numerical conditioning for large European-scale models.

When interpreting results, always check:

- the `SCALE_POWER_TO_GW` setting,
- the power-unit metadata written to the run configuration,
- column names in exported CSV files, and
- whether a post-processing script converts values back to MW or keeps the
  scaled model unit.

Energy-not-served and reserve slack are model quantities. If power is scaled to
GW, reported slack values in model outputs are also in GW unless an export file
or figure explicitly converts them.

## Main Configuration

Most experiments can be controlled from `optimization_tyndp_opf.py`.

### Scenario and Year Selection

Important settings are:

- `YEAR`: target investment or scenario year, such as 2030 or 2040.
- `WEATHER_YEARS`: candidate weather years.
- `WEATHER_YEAR_SELECTION`: selected representative weather years.
- `WEATHER_WEIGHTS_FILE`: file containing weather-year weights.

Maintenance schedules are optimized for the target year. Dispatch and adequacy
are evaluated over the selected weather years.

### Workflow Switches

The main workflow switches are:

- `HEURISTIC`: run the constructive heuristic.
- `BENDERS`: run the Benders decomposition.
- compact MIP mode: used when both `HEURISTIC` and `BENDERS` are disabled.
- `LINE_MAINT`: include transmission maintenance data and variables.
- `FIX_LINE_MAINTENANCE_FROM_HEURISTIC`: fix AC/DC maintenance schedules to a
  previously generated heuristic schedule.
- `WARM_START_HEURISTIC`: initialize thermal maintenance decisions from a
  heuristic schedule.
- `WARM_START_HEURISTIC_DIR`: directory containing the heuristic schedule used
  for warm starts or fixed transmission maintenance.

The recommended publication workflows are described below.

### Network Formulation

Relevant settings include:

- `FLOW_FORMULATION`: selects the network-flow formulation used in the solver.
- `NTC`: activates optional cross-border transfer restrictions if data are
  available.
- `EXACT_SINGLE_LINE_OUTAGE`: uses a more detailed outage representation for
  single-circuit AC corridors.
- `DISAGGREGATE_PARALLEL_AC_LINES`: controls whether parallel AC branches are
  represented explicitly or aggregated.
- `BIG_M_FLOW_FACTOR`: scales big-M constants for outage-related flow-angle
  relaxations.
- `LINE_MAINT_MAX_BORDER_MAINT_CAPACITY_SHARE`: limits the maintained share of
  border transfer capacity in one week.

The exact AC line outage formulation can become computationally demanding. For
large stochastic cases, the current publication workflow avoids optimizing the
full transmission-maintenance topology state directly and instead fixes the TMS
schedule from the heuristic.

### Thermal Maintenance Parameters

The thermal scheduling configuration controls:

- revision duration by fuel and technology,
- long-revision eligibility,
- minimum and maximum long-revision shares,
- maximum parallel revisions per country and week,
- CHP winter restrictions,
- target-year technology mapping,
- minimum capacity thresholds for included units.

Thermal units are grouped where appropriate. The model then schedules group
availability and maintenance starts while retaining enough detail to represent
technology-specific outage durations and reserve eligibility.

### Transmission Maintenance Parameters

Transmission-maintenance settings describe:

- AC corridor maintenance duration,
- DC link maintenance duration,
- annual maintenance frequency per circuit or pole,
- maximum simultaneously maintained units per country and week,
- limits on the maintained capacity share of interconnectors,
- border and country mappings for AC and DC assets.

These settings must be mutually consistent. If required maintenance frequencies
are high but weekly country limits are tight, the scheduling problem may become
infeasible even before considering dispatch.

### Objective Weights

The main adequacy-oriented objective combines:

- expected energy not served,
- frequency-reserve slack,
- self-supply slack,
- the minimum country-week capacity margin, and
- a small tie-breaker for average capacity margins.

The slack penalty must dominate the capacity-margin terms. Otherwise, the model
could prefer a better margin distribution while tolerating load shedding or
reserve violations. The tie-breaker should remain small enough that it only
selects among otherwise similar solutions.

Typical parameters are:

- `CAPACITY_RESERVE_SLACK_PENALTY_M`,
- `COUNTRY_SELF_SUPPLY_TARGET`,
- `COUNTRY_SELF_SUPPLY_SLACK_PENALTY_M`,
- `COUNTRY_SELF_SUPPLY_HARD`, and
- `CAPACITY_RESERVE_MARGIN_TIEBREAK_EPSILON`.

### Gurobi Parameters

Solver settings are passed through `GUROBI_PARAMETERS`. The most relevant
settings are usually:

- time limit,
- MIP gap,
- thread count,
- method,
- presolve,
- numeric focus,
- log output.

For Benders runs, there are additional parameters controlling the number of
iterations, cut selection, subproblem concurrency, and violation tolerances.

## Mathematical Model Overview

The model is a two-stage stochastic maintenance and OPF problem.

### Indices

The main index sets are:

- weather-year scenarios,
- weeks,
- countries,
- nodes or buses,
- thermal generator groups,
- AC corridors,
- DC links,
- storage technologies,
- renewable technologies.

Maintenance decisions are indexed by week and asset. Dispatch decisions are
indexed by scenario, week, and location or asset.

### First-Stage Maintenance Decisions

First-stage decisions are common to all weather years:

- thermal maintenance start week,
- thermal group availability,
- long-revision assignment where enabled,
- AC corridor maintenance start and active outage state,
- DC link maintenance start and active outage state.

These decisions represent the maintenance plan for the target year.

### Second-Stage Recourse Decisions

For each weather year and week, the model evaluates dispatch and network
operation under the fixed maintenance state:

- thermal dispatch,
- hydro storage and other controllable non-RES dispatch,
- battery charge and discharge,
- renewable generation and curtailment,
- demand-side response where available,
- frequency-reserve provision,
- reserve slack,
- energy not served,
- voltage angles,
- AC flows,
- DC flows,
- nodal power balance.

The recourse block is an OPF-style linear dispatch model. It does not model a
full unit-commitment problem with integer on/off states, minimum up/down times,
startup costs, or ramping.

### Thermal Maintenance Constraints

The generator maintenance model enforces:

- one maintenance outage per eligible thermal group,
- technology-specific maintenance duration,
- no starts that would exceed the planning horizon,
- active availability from start-duration convolution,
- country-week limits on parallel maintenance,
- optional long-revision shares,
- CHP winter-start restrictions,
- non-negativity and binary domains where required.

The CHP winter restriction is implemented as a start-week restriction. CHP
maintenance groups are not allowed to start maintenance in predefined winter
weeks. The exact set of winter weeks is controlled by the configuration.

### Transmission Maintenance Constraints

The transmission maintenance model enforces:

- annual maintenance frequency for AC corridors and DC links,
- active outage states from start-duration convolution,
- no starts that would exceed the planning horizon,
- per-asset limits on simultaneously maintained units,
- country-week limits on line maintenance,
- border capacity-share restrictions,
- AC and DC variable domains.

Transmission outages reduce available transfer capacity. For AC lines, they can
also change the effective susceptance in the DC power-flow approximation. The
fully endogenous topology representation is computationally hard at the
considered scale; therefore, the publication workflow can fix line outages from
the heuristic and optimize generator maintenance with that transmission plan.

### Dispatch, Reserve, and Grid Constraints

For every selected weather year and week, the OPF recourse model enforces:

- thermal capacity limits under maintenance availability,
- renewable, hydro, storage, and other non-RES availability limits,
- battery storage charge/discharge and state limits where represented,
- country-level frequency-reserve requirements,
- reserve contribution limits by eligible technology,
- reserve slack accounting,
- nodal power balance,
- AC and DC flow limits,
- optional net transfer capacity restrictions,
- energy-not-served accounting.

The capacity margin used in the objective is based on available thermal
capacity, available hydro-storage support, available other non-RES support,
expected load, and frequency-reserve requirements. It is not a full security
margin and should be interpreted as an adequacy-oriented screening metric.

## Solution Workflows

### Heuristic Workflow

The heuristic is designed to produce a feasible and interpretable maintenance
schedule quickly. It is especially useful for constructing transmission
maintenance schedules that can later be fixed in the optimization model.

The heuristic workflow is:

1. Compute residual-load and adequacy stress indicators by country and week.
2. Build thermal maintenance tickets from grouped generator data.
3. Select long-revision tickets if long maintenance is enabled.
4. Schedule thermal maintenance greedily under country-week limits and CHP
   restrictions.
5. Estimate frequency-reserve slack for the thermal schedule.
6. Sample baseline OPF conditions to score line criticality.
7. Schedule AC and DC maintenance greedily under country and border constraints.
8. Evaluate the full fixed maintenance schedule with OPF recourse.
9. If required, run local repair moves for transmission outages and re-evaluate.
10. Write schedule, evaluation, and diagnostic outputs.

The heuristic is not a proof of optimality. Its purpose is to generate a robust
candidate schedule and a practical transmission outage plan for large instances.

### Compact MIP Workflow

The compact MIP solves the integrated model in one optimization problem. It is
useful for:

- small test cases,
- reduced scenario sets,
- debugging,
- validating heuristic decisions,
- exact fixed-schedule evaluations.

For full stochastic European-scale instances with transmission maintenance
variables, the compact MIP can become too large.

### Benders Decomposition Workflow

The Benders approach separates the master maintenance problem from weekly OPF
recourse subproblems.

The master problem contains:

- generator maintenance starts and availability,
- optional line maintenance starts and outage states,
- reserve slack variables,
- capacity-margin variables,
- self-supply slack variables,
- recourse estimators for expected ENS.

For each scenario and week, a subproblem solves the OPF recourse problem for the
current maintenance state. The subproblem returns the recourse value and dual
information. The master receives cuts that approximate the recourse function.

Cut coefficients are generated from dual multipliers associated with:

- thermal capacity restrictions,
- reserve-related restrictions,
- AC flow restrictions,
- DC flow restrictions,
- line outage state restrictions where active.

The implementation can rank violated cuts and keep only the most relevant ones.
This avoids adding every possible cut when many scenario-week subproblems are
solved in one iteration.

In the current publication setup, the Benders workflow is commonly used with
fixed transmission maintenance from the heuristic. Generator maintenance can be
cold-started or warm-started from the heuristic.

### Fixed-Schedule Evaluation

A fixed schedule can be evaluated with the same OPF recourse equations used in
the optimization model. This is important because heuristic schedules and
optimized schedules should be compared under the same dispatch, reserve, and
network assumptions.

The fixed-schedule evaluation reports:

- expected ENS,
- frequency-reserve slack,
- country-week adequacy indicators,
- network loading,
- critical weeks,
- dispatch by technology,
- cross-border and zone-pair flows where exported.

## Recommended Publication Runs

The following run patterns are recommended for reproducible paper results.

### 1. Heuristic Schedule and Evaluation

Use this to generate a feasible thermal and transmission maintenance schedule.

```python
HEURISTIC = True
BENDERS = False
WARM_START_HEURISTIC = False
FIX_LINE_MAINTENANCE_FROM_HEURISTIC = False
```

This run produces the schedule files that can later be used for fixed TMS
optimization.

### 2. Benders With Fixed Transmission Maintenance

Use this when the heuristic transmission schedule should be fixed and generator
maintenance should be optimized.

```python
HEURISTIC = False
BENDERS = True
FIX_LINE_MAINTENANCE_FROM_HEURISTIC = True
WARM_START_HEURISTIC_DIR = r"path\to\heuristic\run"
WARM_START_HEURISTIC = False
```

This is a cold-start generator maintenance run under fixed TMS.

### 3. Benders With Fixed Transmission Maintenance and Thermal Warm Start

Use this when the heuristic should also provide an initial generator schedule.

```python
HEURISTIC = False
BENDERS = True
FIX_LINE_MAINTENANCE_FROM_HEURISTIC = True
WARM_START_HEURISTIC_DIR = r"path\to\heuristic\run"
WARM_START_HEURISTIC = True
```

This run keeps the heuristic TMS schedule fixed and initializes the GMS decision
variables from the heuristic. The optimizer may still change generator
maintenance decisions unless those variables are explicitly fixed.

### 4. Compact MIP

Use this only for smaller instances or controlled tests.

```python
HEURISTIC = False
BENDERS = False
```

For full-size stochastic instances with line maintenance, this workflow can be
computationally expensive.

## Installation

The repository contains a conda environment file:

```powershell
conda env create -f environment.maint-model.yml
conda activate maint-model
```

The environment uses:

- Python 3.11,
- NumPy,
- pandas,
- scikit-learn,
- xarray,
- netCDF4,
- Gurobi 12.

A working Gurobi installation and license are required.

## Running the Model

From this directory:

```powershell
python optimization_tyndp_opf.py
```

Before running, check at least:

1. `DIR_BASE` points to the prepared input data.
2. `DIR_OUT` points to a writable output directory.
3. `YEAR` matches the input data.
4. `WEATHER_YEAR_SELECTION` and `WEATHER_WEIGHTS_FILE` are consistent.
5. `HEURISTIC`, `BENDERS`, `WARM_START_HEURISTIC`, and
   `FIX_LINE_MAINTENANCE_FROM_HEURISTIC` describe the intended workflow.
6. `LINE_MAINT` matches the intended treatment of transmission maintenance.
7. Gurobi time limit and MIP gap are appropriate for the instance size.

The run script creates a timestamped output folder under the configured output
root.

## Output Structure

Outputs are written to a target-year and timestamp-specific run directory,
typically:

```text
<DIR_OUT>/<YEAR>/<RUN_ID>/
```

Common output categories are:

- run configuration and metadata,
- solver status and phase timings,
- maintenance schedules,
- dispatch results,
- reserve and adequacy results,
- grid-flow results,
- critical-week and stress indicators,
- Benders iteration logs,
- Benders cut diagnostics,
- heuristic repair logs,
- fixed-schedule evaluation summaries.

The exact file names depend on the selected workflow. The most important files
for publication comparisons are usually:

- run configuration JSON,
- phase-time CSV files,
- thermal maintenance schedule CSV files,
- AC maintenance schedule CSV files,
- DC maintenance schedule CSV files,
- exact fixed-schedule summary files,
- weekly adequacy result files,
- Benders iteration and cut logs,
- heuristic schedule and repair diagnostics.

## Interpreting Results

### Energy Not Served and Reserve Slack

Energy not served and frequency-reserve slack are penalty quantities. They
should normally be interpreted together. A solution with lower ENS but much
higher reserve slack may not be operationally preferable, depending on the
objective weights and publication metric.

### Capacity Margin

The country-week capacity margin compares available capacity against expected
load and reserve requirements. It includes:

- available thermal capacity after maintenance,
- available hydro-storage support,
- available other non-RES support,
- expected load,
- reserve requirement.

It does not fully represent dynamic stability, N-1 security, ramping, or
unit-commitment feasibility.

### Benders Gap

The Benders gap reports convergence of the decomposition workflow. If the run
terminates early, the incumbent solution can still be useful, but the reported
gap should be included when comparing heuristic and optimization results.

When transmission maintenance is fixed from the heuristic, the Benders run is
not optimizing the full integrated GMS-TMS problem. It is optimizing generator
maintenance and recourse decisions under the fixed line maintenance schedule.

### Runtime

Runtime depends strongly on:

- number of selected weather years,
- number of weeks,
- reduced grid size,
- line-maintenance formulation,
- whether TMS is fixed,
- Gurobi parameters,
- Benders cut-management settings,
- number of subproblem workers,
- enabled output detail.

Comparisons between heuristic and Benders runs should state whether TMS was
optimized, fixed, or excluded.

## Reproducibility

Each run should be documented with:

- git commit hash,
- target year,
- selected weather years,
- weather-year weights,
- scenario reduction method,
- input data version,
- output directory,
- solver version,
- Gurobi parameters,
- workflow flags,
- objective weights,
- line-maintenance assumptions,
- whether TMS was fixed from the heuristic,
- whether generator maintenance was warm-started.

The run configuration file written by the script is the primary record of these
settings.

For publication, it is recommended to archive:

- the exact run configuration,
- selected processed input files if redistribution is allowed,
- maintenance schedules,
- summary result tables,
- scripts used for figure generation,
- solver logs for the reported runs.

Large raw data files should only be published if licensing and data-provider
terms allow it.

## Common Pitfalls

### Weather Weights Do Not Match Selected Weather Years

The selected weather years and the weight file must describe the same scenario
set. If a selected year is missing or a weight is assigned to an unused year,
expected-load and expected-ENS results become inconsistent.

### MW and GW Are Mixed

If `SCALE_POWER_TO_GW` is enabled, the model solves in scaled units. Check
metadata and export labels before using values in tables or figures.

### Line Maintenance Is Infeasible

High annual maintenance frequencies, long durations, and strict weekly country
limits can make TMS infeasible. Relax country-week limits or reduce maintenance
frequency before debugging the OPF.

### PTDF Formulation With Endogenous Line Outages

If transmission maintenance changes the network topology, a fixed PTDF
representation can become inconsistent. Use the theta-based formulation for
line-maintenance experiments unless the topology is fixed.

### Fixed TMS Requires a Valid Heuristic Directory

When `FIX_LINE_MAINTENANCE_FROM_HEURISTIC` is enabled, the warm-start directory
must contain compatible AC and DC maintenance schedule files for the same year,
same asset identifiers, and same line-maintenance settings.

### Self-Supply Constraint Can Cause Infeasibility

If the country self-supply requirement is enforced as a hard constraint, some
country-week combinations may become infeasible. The publication workflow
usually treats this through a slack penalty unless a hard feasibility test is
intended.

### Benders Cuts Can Become Too Numerous

Large scenario-week sets can generate many cuts. Use cut selection parameters
to keep only strongly violated or hard-violation cuts. Otherwise, master
problem size can grow quickly.

## Extending the Model

Possible extensions include:

- alternative weather-year clustering and weighting,
- additional maintenance duration assumptions,
- more detailed reserve eligibility rules,
- explicit N-1 screening after schedule generation,
- alternative country-level adequacy metrics,
- higher temporal resolution,
- more detailed hydro and storage operation,
- improved decomposition stabilization,
- stronger transmission maintenance valid inequalities,
- publication-specific post-processing and figure generation.

Any extension that changes the mathematical problem should also update:

- the run configuration export,
- output metadata,
- README documentation,
- paper nomenclature,
- paper equations,
- post-processing scripts.

## Limitations

The model has several intentional simplifications:

- weekly temporal resolution,
- reduced European grid representation,
- DC power-flow approximation,
- no full unit-commitment representation,
- no explicit startup or shutdown costs,
- no minimum up/down constraints,
- no dynamic stability constraints,
- no endogenous N-1 security enforcement in the current publication workflow,
- fixed transmission maintenance in the recommended Benders publication setup,
- stochasticity represented through selected weather years and weights.

These limitations should be stated when interpreting results. The model is best
understood as an adequacy- and network-aware maintenance planning model, not as
a full operational security-constrained unit-commitment tool.

## Data and Licensing Notes

The workflow can use data derived from public or semi-public European power
system sources, including TYNDP datasets and ENTSO-E-derived time series.
Redistribution rights depend on the original data source and the processing
pipeline.

