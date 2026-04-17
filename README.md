# TaSe2 Phase Switching Analyzer

Desktop analysis tooling for TaSe2 phase-switching datasets, with the notebook workflow extracted into a reusable Python pipeline and a desktop UI for interactive exploration.

## What the app does

- Loads 1 to 4 NetCDF datasets in sequence order.
- Detects the cross-shaped contact region automatically from the spatial intensity maps.
- Extracts per-pixel spectral features near the Fermi level and across the full local spectrum.
- Runs shared PCA and k-means clustering so you can compare spectral classes across multiple pulse states.
- Builds simple state maps and per-pixel sequence maps to show how pixels switch between states over time.
- Lets you click any pixel to inspect its local spectrum across the full sequence.
- Exports the analysis outputs as `.json` summaries and `.npy` arrays.

## Launch the desktop app

### Option 1: run from the repository

```bash
python run_desktop_app.py
```

You can also preload files from the command line:

```bash
python run_desktop_app.py /path/to/state0.nc /path/to/state1.nc /path/to/state2.nc
```

### Option 2: install the package and use the script entry point

```bash
pip install -e .
tase2-desktop
```

### macOS shortcut

You can also double-click `launch_tase2_app.command` after dependencies are installed.

## Install dependencies

```bash
pip install -e .
```

The app uses a NumPy-based PCA and k-means implementation, so it does not depend on `torch` to run the desktop workflow.

## Run the smoke test

```bash
python -m unittest tests.test_analysis_pipeline
```

## Build a standalone executable

One straightforward path is PyInstaller:

```bash
pip install pyinstaller
pyinstaller --windowed --name TaSe2Analyzer run_desktop_app.py
```

That creates a bundled executable in `dist/`.

## Repository layout

- `src/tase2_phase_switching/analysis.py`: reusable notebook-derived analysis pipeline.
- `src/tase2_phase_switching/desktop_app.py`: desktop UI built with `tkinter` and embedded Matplotlib.
- `src/analysis/clustering_pipeline.ipynb`: original notebook reference for the clustering and sequence workflow.
- `src/analysis/data_processing..ipynb`: original dataset inspection notebook.
