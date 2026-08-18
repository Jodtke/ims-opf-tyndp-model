# Runbook: Wetterjahresvarianten vollständig neu rechnen

Dieses Runbook beschreibt die reproduzierbare Berechnung der ursprünglichen
Kalenderjahresvariante (`jan_dec`) und der verschobenen Wartungsjahresvariante
(`w17_w16`). Es ist für Windows mit Git Bash ausgelegt und kann nach einem
`git pull` auf einem anderen Rechner verwendet werden.

Die Pipeline umfasst jeweils:

1. Erzeugung der reinen, stressbasierten Wartungsheuristik ohne Repair,
2. Fixed-Schedule-OPF-Evaluation der Heuristik,
3. Kopieren der Heuristikpläne in die szenariospezifischen Warm-Start-Ordner,
4. monolithische MIP-Optimierung der GMS bei fixierter heuristischer TMS.

In Schritt 4 bleibt die thermische GMS vollständig optimierbar. Der
Matrix-Runner verwendet die heuristische GMS zusätzlich als MIP-Start, fixiert
aber ausschließlich die AC/DC-Wartungspläne. Dies ist der im Paper verwendete
Fixed-TMS-Workflow.

Optional ersetzt `--sequential-tms-gms` Schritt 4 für die nodalen OPF-Fälle
durch zwei aufeinanderfolgende Optimierungen: zuerst TMS bei fixierter
heuristischer GMS, danach GMS bei fixierter optimierter TMS. `ed_national`
bleibt unverändert, da dort keine TMS modelliert wird. Der Schalter funktioniert
sowohl mit dem monolithischen MIP als auch mit optional aktiviertem Benders.

Repair-Heuristiken und Cold-Start-Optimierungen sind bewusst nicht Bestandteil
dieser Pipeline. Die Befehle dafür sind am Ende separat dokumentiert.

## 1. Berechnete Szenarien

### Wetterjahresprofile

| Profil | Modellwoche 1 | Abgedeckter Zeitraum |
|---|---:|---|
| `jan_dec` | Kalenderwoche 1 | Kalenderwoche 1 bis 52 desselben Jahres |
| `w17_w16` | Kalenderwoche 17 | Kalenderwoche 17 bis Woche 16 des Folgejahres |

Für die TYNDP-Läufe verwendet `jan_dec` die Wetterjahre 1982 bis 2016 als
Grundgesamtheit. Bei `w17_w16` sind die Startjahre 1982 bis 2015, weil für jede
52-Wochen-Sequenz auch die Wochen 1 bis 16 des Folgejahres benötigt werden.
Beide Varianten verwenden in diesem Runbook die jeweilige `k07`-Reduktion.

Für das Actual-2025-Szenario werden die Quelldaten 2016 bis 2025 verwendet:

- `jan_dec`: zehn Kalenderjahre 2016 bis 2025,
- `w17_w16`: neun Wartungsjahre mit Startjahren 2016 bis 2024; das Jahr 2025
  liefert die abschließenden Wochen 1 bis 16.

### Zieljahre und Netzmodelle

| Datenbasis | Zieljahre | Netzvarianten |
|---|---|---|
| TYNDP-Matrix | 2030 und 2040 | `k128`, `k256`, `ed_national` |
| Actual-2025 | 2025 | `k128`, `k256`, `ed_national` |

`ed_national` aggregiert Erzeugung, Last und Flexibilität auf Länderknoten. Die
Grenzkuppelkapazitäten stammen standardmäßig aus aggregierten AC/DC-Leitungen
(`line_aggregate`). In diesem Modus wird keine Transmission Maintenance
Scheduling (TMS) durchgeführt.

### Export-Shortage-Varianten

Jede Fixed-Schedule-Evaluation und jede Warm-Start-Optimierung wird zweimal
gerechnet:

- `export_guard`: Ein Land darf bei eigener ENS nicht netto exportieren. Der
  FR-Bedarf ist bereits als nationaler Lastaufschlag enthalten.
- `no_export_guard`: Gleichzeitiger Nettoexport und nationale Knappheit sind
  zugelassen.

Der Export-Shortage-Guard ist **kein vollständiges Exportverbot**. Er verhindert
Exporte nur bei gleichzeitiger Knappheit im exportierenden Land.

Die reine Schedule-Only-Heuristik hängt nicht von diesem Guard ab. Deshalb wird
sie je Wetterjahresprofil nur einmal berechnet und anschließend für beide
Varianten verwendet. Evaluationsergebnisse und Warm-Start-Verzeichnisse bleiben
durch unterschiedliche Suffixe beziehungsweise Namespaces getrennt.

## 2. Verwendete Modellkonfiguration

Die tatsächlichen Defaults in `model/optimization_tyndp_opf.py` sind:

```python
NETWORK_MODE = "opf"
FLOW_FORMULATION = "theta"
COUNTRY_EXPORT_SHORTAGE_GUARD = True
WARM_START_HEURISTIC = False
FIX_LINE_MAINTENANCE_FROM_HEURISTIC = False
```

Die Matrix-Runner überschreiben Defaults abhängig von Stage und CLI-Schaltern.
Insbesondere aktiviert die Stage `warm-optimizations` den heuristischen
MIP-Start. Dieses Runbook ergänzt
`--fix-line-maintenance-from-heuristic`, damit der Paper-Lauf die
heuristische TMS fixiert und nur die GMS optimiert. Die effektive Konfiguration
jedes Laufs steht in `run_config.json`.

Die Befehle setzen beziehungsweise übernehmen außerdem:

- Zielfunktion: `ens`, also Minimierung der wettergewichteten erwarteten ENS,
- Optimierungsverfahren: monolithisches MIP; Benders nur bei expliziter Wahl,
- AC/DC-Belastungsgrenze: standardmäßig 70 % der Nennkapazität,
- lange thermische Revisionen: deaktiviert,
- TMS: aktiv für die nodalen `k128`- und `k256`-OPF-Netze,
- exakte Single-Line-Ausfalllogik: aktiv bei TMS,
- einzelne AC-Verbindungen zwischen einem Buspaar: von der Wartung ausgenommen,
- nationale ED-Grenzkapazitäten: `line_aggregate`,
- Repair-Heuristik: deaktiviert,
- Cold-Start-Optimierung: nicht Teil des Standardablaufs.

Die Single-AC-Ausnahme wird in
`EXEMPT_SINGLE_AC_CONNECTIONS_FROM_MAINTENANCE=True` abgebildet. Jeder Lauf
schreibt dazu `opf_ac_maintenance_requirements.csv` in sein Ergebnisverzeichnis.

## 3. Voraussetzungen auf einem neuen Rechner

### Repository aktualisieren

```bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git maintenance_model
cd maintenance_model
git pull --ff-only origin main
```

Wenn das Repository bereits vorhanden ist:

```bash
cd /absolute/path/to/maintenance_model
git pull --ff-only origin main
```

### Conda-Umgebung und Gurobi

Die vorhandene Umgebung kann direkt aktiviert werden:

```bash
conda activate maint-model
python --version
python -c "import gurobipy, pandas; print('Python dependencies available')"
```

Falls die Umgebung auf dem neuen Rechner noch nicht existiert:

```bash
conda env create -n maint-model -f environment.yaml
conda activate maint-model
```

Für eine exakt aufgelöste Windows-Umgebung kann stattdessen die geprüfte
`environment.win-64.lock.yml` verwendet werden:

```bash
conda env create -f environment.win-64.lock.yml
conda activate maint-model
```

Zusätzlich muss eine gültige Gurobi-Lizenz verfügbar sein. Bei Problemen sollte
vor dem Matrixlauf ein kleines Gurobi-Modell oder mindestens der Import von
`gurobipy` getestet werden.

Falls `conda activate` in Git Bash nicht verfügbar ist, kann Conda einmalig in
die Shell eingebunden werden:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maint-model
```

Alternativ kann `PYTHON` vor dem Start auf die konkrete `python.exe` gesetzt
werden:

```bash
export PYTHON='/absolute/path/to/conda/envs/maint-model/python'
```

### Eingabedaten

Die Eingabe- und Ausgabedaten liegen außerhalb des Repositories. Vor dem Lauf
werden portable absolute Pfade gesetzt:

```bash
export INPUT_ROOT='/absolute/path/to/revision_outage_input'
export OUTPUT_ROOT='/absolute/path/to/revision_outage_output'
```

Die vollständige Verzeichnisstruktur und die benötigten Netz-, Last-,
Erzeugungs-, Wetter- und Wartungsdateien sind im Root-`README.md` beschrieben.
Der Matrix-Runner erzeugt `$INPUT_ROOT/warm_start` während der Pipeline.

Für die verschobene TYNDP-Variante müssen insbesondere folgende Dateien bereits
vorliegen:

```text
<INPUT_ROOT>/weather_year_reduction/scenarios/w17_w16/target_year_2030/source_weather_week_schedule.csv
<INPUT_ROOT>/weather_year_reduction/scenarios/w17_w16/target_year_2030/k07/weather_year_selection_target_year_2030_k07.csv
<INPUT_ROOT>/weather_year_reduction/scenarios/w17_w16/target_year_2030/k07/weatherYears_weights_reduced_target_year_2030_k07.csv
<INPUT_ROOT>/weather_year_reduction/scenarios/w17_w16/target_year_2040/source_weather_week_schedule.csv
<INPUT_ROOT>/weather_year_reduction/scenarios/w17_w16/target_year_2040/k07/weather_year_selection_target_year_2040_k07.csv
<INPUT_ROOT>/weather_year_reduction/scenarios/w17_w16/target_year_2040/k07/weatherYears_weights_reduced_target_year_2040_k07.csv
```

Der Matrix-Runner nutzt diese vorhandenen Wetterjahresreduktionen. Er berechnet
die TYNDP-Clusterung nicht erneut. Der Actual-2025-Runner verwendet vorhandene
Gewichte; fehlen sie, erzeugt er für die gültigen Wartungsjahr-Startjahre eine
gleichgewichtete Datei.

## 4. Vollständiges Git-Bash-Skript

Das folgende Skript wird aus dem Repository-Root ausgeführt. `INPUT_ROOT` muss
gesetzt sein. `OUTPUT_ROOT` ist optional und fällt auf
`<repository>/output` zurück. `TYNDP_OUTPUT_ROOT` und
`ACTUAL_OUTPUT_ROOT` können bei Bedarf separat überschrieben werden.

`RUN_TAG` identifiziert die gesamte Ausführung. Für einen neuen vollständigen
Lauf wird automatisch ein Zeitstempel verwendet. Für die Wiederaufnahme eines
abgebrochenen Laufs muss derselbe zuvor ausgegebene `RUN_TAG` gesetzt werden.

```bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-weather_rerun_$(date +%Y%m%d_%H%M%S)}"
SEQUENTIAL_TMS_GMS="${SEQUENTIAL_TMS_GMS:-0}"

SEQUENTIAL_ARGS=()
if [[ "$SEQUENTIAL_TMS_GMS" == "1" ]]; then
    SEQUENTIAL_ARGS=(--sequential-tms-gms)
fi

: "${INPUT_ROOT:?Set INPUT_ROOT to the absolute external input directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/output}"
TYNDP_OUTPUT_ROOT="${TYNDP_OUTPUT_ROOT:-$OUTPUT_ROOT/opf_tyndp2024}"
ACTUAL_OUTPUT_ROOT="${ACTUAL_OUTPUT_ROOT:-$OUTPUT_ROOT/opf_actual_2025}"
mkdir -p "$TYNDP_OUTPUT_ROOT" "$ACTUAL_OUTPUT_ROOT"

printf 'RUN_TAG=%s\n' "$RUN_TAG"
printf 'SEQUENTIAL_TMS_GMS=%s\n' "$SEQUENTIAL_TMS_GMS"
"$PYTHON" --version

run_tyndp_profile() {
    local profile="$1"
    local tag="${RUN_TAG}_tyndp_${profile}"
    local heuristic_batch="${tag}_heuristic"
    local common=(
        --years 2030 2040
        --weather k07
        --models k128 k256
        --network-modes opf ed_national
        --maintenance-year-profiles "$profile"
        --national-capacity-source line_aggregate
        --national-resource-model k256
        --objective-preset ens
        --dir-base "$INPUT_ROOT"
        --dir-out "$TYNDP_OUTPUT_ROOT"
    )

    "$PYTHON" model/run_tyndp2024_matrix.py \
        --stage pure-heuristics "${common[@]}" \
        --batch-id "$heuristic_batch"

    local mode guard_flag
    for mode in export_guard no_export_guard; do
        if [[ "$mode" == "export_guard" ]]; then
            guard_flag="--country-export-shortage-guard"
        else
            guard_flag="--no-country-export-shortage-guard"
        fi

        "$PYTHON" model/run_tyndp2024_matrix.py \
            --stage evaluate-heuristics "${common[@]}" "$guard_flag" \
            --batch-id "$heuristic_batch" \
            --heuristic-evaluation-suffix "_heuristic_eval_${mode}"

        "$PYTHON" model/run_tyndp2024_matrix.py \
            --stage prepare-warm-start "${common[@]}" "$guard_flag" \
            --batch-id "$heuristic_batch" \
            --warm-start-namespace "$mode"

        "$PYTHON" model/run_tyndp2024_matrix.py \
            --stage warm-optimizations "${common[@]}" "${SEQUENTIAL_ARGS[@]}" "$guard_flag" \
            --fix-line-maintenance-from-heuristic \
            --batch-id "${tag}_${mode}" \
            --warm-start-namespace "$mode"
    done
}

run_actual_profile() {
    local profile="$1"
    local tag="${RUN_TAG}_actual_${profile}"
    local heuristic_batch="${tag}_heuristic"
    local common=(
        --models k128 k256
        --network-modes opf ed_national
        --maintenance-year-profiles "$profile"
        --weather-years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
        --national-capacity-source line_aggregate
        --national-resource-model k256
        --workflow mip
        --run-kinds warm
        --objective-preset ens
        --line-maint
        --no-long-revisions
        --input-root "$INPUT_ROOT"
        --output-root "$ACTUAL_OUTPUT_ROOT"
    )

    "$PYTHON" model/run_actual_2025_matrix.py \
        --stage pure-heuristics "${common[@]}" \
        --batch-id "$heuristic_batch"

    local mode guard_flag
    for mode in export_guard no_export_guard; do
        if [[ "$mode" == "export_guard" ]]; then
            guard_flag="--country-export-shortage-guard"
        else
            guard_flag="--no-country-export-shortage-guard"
        fi

        "$PYTHON" model/run_actual_2025_matrix.py \
            --stage evaluate-heuristics "${common[@]}" "$guard_flag" \
            --batch-id "$heuristic_batch" \
            --heuristic-evaluation-suffix "_heuristic_eval_${mode}"

        "$PYTHON" model/run_actual_2025_matrix.py \
            --stage prepare-warm-start "${common[@]}" "$guard_flag" \
            --batch-id "$heuristic_batch" \
            --warm-start-namespace "$mode"

        "$PYTHON" model/run_actual_2025_matrix.py \
            --stage warm-optimizations "${common[@]}" "${SEQUENTIAL_ARGS[@]}" "$guard_flag" \
            --fix-line-maintenance-from-heuristic \
            --batch-id "${tag}_${mode}" \
            --warm-start-namespace "$mode"
    done
}

# Zuerst die ursprüngliche Kalenderjahresvariante vollständig rechnen.
run_tyndp_profile jan_dec
run_actual_profile jan_dec

# Danach die verschobene Wartungsjahresvariante vollständig rechnen.
run_tyndp_profile w17_w16
run_actual_profile w17_w16
```

Der Fixed-TMS-Paper-Ablauf bleibt mit `SEQUENTIAL_TMS_GMS=0` der Default.
Für die alternative sequenzielle TMS-dann-GMS-Pipeline wird vor dem Start
gesetzt:

```bash
export SEQUENTIAL_TMS_GMS=1
```

Je nodalem OPF-Szenario entstehen dann getrennte Run-Verzeichnisse mit
`seqtms_fixedgms` und `seqgms_fixedtms` im Namen. Der kombinierte Eingang für
die zweite Optimierung liegt unter `<TMS_RUN>/gms_warm_start` und enthält ein
`sequential_warm_start_source.json` mit der Herkunft beider Teilpläne. Da pro
OPF-Fall zwei vollwertige Optimierungen laufen, kann sich dessen maximale
Solverzeit gegenüber dem bisherigen Warm-Start-Lauf annähernd verdoppeln.

## 5. Reihenfolge und Anzahl der Läufe

Das Skript läuft strikt sequenziell, damit nicht mehrere große Gurobi-Modelle
gleichzeitig um CPU und Arbeitsspeicher konkurrieren.

Je TYNDP-Profil werden sechs Szenarien verarbeitet:

```text
2030-k128, 2030-k256, 2030-ed_national,
2040-k128, 2040-k256, 2040-ed_national
```

Je Actual-2025-Profil werden drei Szenarien verarbeitet:

```text
2025-k128, 2025-k256, 2025-ed_national
```

Die Heuristik wird je Profil und Datenbasis einmal berechnet. Evaluation und
Warm-Optimierung werden anschließend jeweils für `export_guard` und
`no_export_guard` durchgeführt. Ohne sequenzielle TMS/GMS-Stufe entstehen über
beide Wetterjahresprofile 36 Warm-Start-Optimierungen:

```text
2 Profile x 2 Guard-Varianten x (6 TYNDP + 3 Actual-2025) = 36
```

Mit `SEQUENTIAL_TMS_GMS=1` wird jeder der sechs nodalen OPF-Fälle zweimal und
jeder der drei nationalen ED-Fälle weiterhin einmal optimiert. Damit entstehen
60 Solverläufe:

```text
2 Profile x 2 Guard-Varianten x ((4 x 2 + 2) TYNDP + (2 x 2 + 1) Actual-2025) = 60
```

## 6. Ausgabe- und Warm-Start-Struktur

Die TYNDP-Ergebnisse werden nach Profil und Zieljahr abgelegt:

```text
<TYNDP_OUTPUT_ROOT>/scenarios/jan_dec/2030/...
<TYNDP_OUTPUT_ROOT>/scenarios/jan_dec/2040/...
<TYNDP_OUTPUT_ROOT>/scenarios/w17_w16/2030/...
<TYNDP_OUTPUT_ROOT>/scenarios/w17_w16/2040/...
```

Die Actual-2025-Ergebnisse liegen unter:

```text
<ACTUAL_OUTPUT_ROOT>/scenarios/jan_dec/...
<ACTUAL_OUTPUT_ROOT>/scenarios/w17_w16/...
```

Die kopierten Warm Starts werden ebenfalls nach Profil, Zieljahr, Netzmodell,
Wetterfall und Guard-Variante getrennt. Typische Pfade sind:

```text
<INPUT_ROOT>/warm_start/scenarios/jan_dec/target_year_2040/<network>/k07/export_guard
<INPUT_ROOT>/warm_start/scenarios/jan_dec/target_year_2040/<network>/k07/no_export_guard
<INPUT_ROOT>/warm_start/scenarios/w17_w16/target_year_2040/<network>/k07/export_guard
<INPUT_ROOT>/warm_start/scenarios/w17_w16/target_year_2040/<network>/k07/no_export_guard
```

Für jeden Lauf sollten mindestens folgende Diagnoseinformationen geprüft werden:

- `run_config.json`: vollständige Parametrisierung,
- `phase_times.csv`: Laufzeiten der Vorverarbeitung und Lösung,
- `opf_maintenance_year_week_mapping.csv`: Zuordnung Modellwoche zu Kalenderwoche,
- `opf_ac_maintenance_requirements.csv`: AC-Wartungspflichten und Ausnahmen,
- Heuristik- und Solver-Status in den jeweiligen Ergebnisdateien.

## 7. Vorabprüfung ohne vollständigen Solverlauf

Vor einem langen Lauf sollten Tests und ein Dry Run ausgeführt werden:

```bash
python -m unittest discover -s tests -v

python model/run_tyndp2024_matrix.py \
    --stage pure-heuristics \
    --years 2030 2040 \
    --weather k07 \
    --models k128 k256 \
    --network-modes opf ed_national \
    --maintenance-year-profiles jan_dec w17_w16 \
    --dir-base "$INPUT_ROOT" \
    --dir-out "$TYNDP_OUTPUT_ROOT" \
    --dry-run

python model/run_actual_2025_matrix.py \
    --stage pure-heuristics \
    --models k128 k256 \
    --network-modes opf ed_national \
    --maintenance-year-profiles jan_dec w17_w16 \
    --input-root "$INPUT_ROOT" \
    --output-root "$ACTUAL_OUTPUT_ROOT" \
    --dry-run
```

Beim Actual-2025-Runner sollte für einen Dry Run nur eine einzelne Stufe wie
`pure-heuristics` verwendet werden. `--stage all --dry-run` erzeugt keine echten
Heuristikdateien, die direkt anschließende simulierte Evaluation könnte diese
daher nicht finden.

## 8. Wiederaufnahme und erneute Ausführung

Der zu Beginn ausgegebene `RUN_TAG` sollte im Laufprotokoll festgehalten werden.
Ein neuer Zeitstempel erzeugt vollständig neue Ergebnisverzeichnisse.

Für eine Wiederaufnahme in einer neuen Git-Bash-Sitzung wird der alte Tag vor
dem erneuten Aufruf gesetzt:

```bash
export RUN_TAG='weather_rerun_YYYYMMDD_HHMMSS'
```

Vollständig abgeschlossene Run-Verzeichnisse werden bei gleichem Batch-ID
übersprungen. Existiert ein unvollständiges Verzeichnis, bricht der Runner
absichtlich ab. Dann bestehen zwei Möglichkeiten:

1. Einen neuen `RUN_TAG` verwenden und den betroffenen Lauf sauber neu starten.
2. Den einzelnen Stage-Befehl kontrolliert mit `--rerun-existing` wiederholen.

`--rerun-existing` sollte nicht pauschal in das vollständige Skript aufgenommen
werden, weil dadurch vorhandene Ergebnisdateien ergänzt oder überschrieben
werden können.

## 9. Optionale Varianten

### Alle Wetterjahre zusätzlich zu k07

Im TYNDP-Block kann

```bash
--weather k07
```

durch

```bash
--weather k07 all
```

ersetzt werden. Dadurch erhöht sich die Zahl und Größe der Modelle erheblich.

### NTC statt aggregierter Leitungskapazitäten im nationalen ED

```bash
--national-capacity-source ntc
```

Bei NTC-Kapazitäten wird der 70-%-Leitungsfaktor nicht auf die NTC-Werte
angewendet.

### Benders statt monolithischem MIP

Für die TYNDP-Matrix wird ausschließlich der Runner-Schalter `--benders`
ergänzt. Beim Actual-2025-Runner wird die Workflow-Auswahl verwendet:

```bash
--workflow benders
```

Der Actual-2025-Runner akzeptiert `--benders` zusätzlich als Alias. Bei einem
direkten Einzelaufruf über `optimization_tyndp_opf.py` wird stattdessen
`BENDERS=True` in der Konfiguration gesetzt. Ohne diese expliziten Angaben
verwenden beide Matrix-Runner das monolithische MIP.

### Cold-Start-Vergleich zusätzlich rechnen

TYNDP:

```bash
python model/run_tyndp2024_matrix.py --stage cold-optimizations <sonstige Argumente>
```

Actual-2025:

```bash
python model/run_actual_2025_matrix.py \
    --stage cold-optimizations --run-kinds cold <sonstige Argumente>
```

### Repair-Heuristik zusätzlich rechnen

Nur der TYNDP-Runner besitzt derzeit eine separate Repair-Stufe:

```bash
python model/run_tyndp2024_matrix.py --stage repair-heuristics <sonstige Argumente>
```

Die in diesem Runbook verwendeten Warm Starts stammen bewusst aus der reinen
Heuristik und nicht aus der Repair-Heuristik.

## 10. Publikationsgrafiken erzeugen

Nach Abschluss aller Runs wird der gemeinsame Output-Root für die
Visualisierung gesetzt. Der folgende Aufruf erzeugt die Hauptgrafiken für
Actual 2025 sowie TYNDP 2030/2040, jeweils für Jan-Dec und Apr-Apr, als PDF:

```bash
export REVISION_OUTAGE_OUTPUT="$OUTPUT_ROOT"

"$PYTHON" visualisation/build_modular_publication_figures.py \
    --dataset all \
    --profiles jan_dec w17_w16 \
    --formats pdf
```

Die Ergebnisordner werden unterhalb von
`$TYNDP_OUTPUT_ROOT/publication_figures` und
`$ACTUAL_OUTPUT_ROOT/publication_figures` angelegt. Die
Visualisierungsskripte für historische Vergleiche und Basisgrafiken benötigen
zusätzliche externe Datensätze; ihre Pfade werden explizit über die jeweiligen
CLI-Argumente gesetzt:

```bash
"$PYTHON" visualisation/build_historical_maintenance_comparison_figures.py --help
"$PYTHON" visualisation/build_thermal_maintenance_duration_figures.py --help
"$PYTHON" visualisation/build_transmission_maintenance_effort_figures.py --help
```
