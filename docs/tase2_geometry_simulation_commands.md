# `tase2_geometry_simulation.py` Terminal Command Reference

This document describes the terminal commands that can be used with `tase2_geometry_simulation.py`, what each command does, and what files and console output it produces. The script is designed to be run from `src/analysis`, while the nanoARPES `.nc` datasets are expected to live in `data/` by default. The script uses file `a` as the reference state unless explicitly overridden, and it simulates geometry-driven local spectral changes together with a region-wise current-response model for a selected target case.

The script accepts either the canonical case names `a`, `b`, `c`, `c2`, and `d`, or explicit file paths supplied through `--base-file` and `--target-file`. The default file names are

`a_convert_2_nosm.nc`, `b_convert_2_nosm.nc`, `c_convert_2_nosm.nc`, `c2_convert_2_nosm.nc`, and `d_convert_2_nosm.nc`.

The most basic invocation is

```bash
python tase2_geometry_simulation.py --target-case b
```

This command loads `a_convert_2_nosm.nc` from `../../data` as the base state and `b_convert_2_nosm.nc` as the target state. It infers a valid device mask from the integrated intensity maps, extracts per-pixel spectral features from the base and target scans, infers a region map from the base state, estimates a local tilt and current-response parameter set for each region, warps the spectra within each region according to those inferred tilt parameters, mixes the warped base cube toward the warped target cube according to the inferred activation map, and writes the simulated cube to a dated output directory. It also renders a summary figure showing the inferred region map, activation map, and near-
`E_F` comparisons.

The console output from this command is progress-oriented rather than silent. It prints timestamped status messages while it loads files, builds the valid mask, extracts features, infers regions, warps the base and target cubes region by region, builds the simulated cube, renders the figure, and writes the output files. At the end of the run, it prints the resolved base file path, target file path, output NetCDF path, summary figure path, and the inferred region parameters for each region, including the `phi` shift, energy shift, and current-mixing value.

A slightly more complete basic run is

```bash
python tase2_geometry_simulation.py --target-case b --save-region-products
```

This performs the same simulation as the previous command but additionally saves the inferred region map, valid mask, and activation map as standalone `.npy` files in the dated run directory. This is the most useful command for iterative work, because it gives direct access to the inferred region partition and the current-response map for later reuse or manual modification.

The script can also be run for any of the other canonical pulse states. The corresponding commands are

```bash
python tase2_geometry_simulation.py --target-case c
python tase2_geometry_simulation.py --target-case c2
python tase2_geometry_simulation.py --target-case d
```

These commands keep `a` as the reference state and change only the target template. The `c` command simulates the state after the orthogonal pulse associated with file `c`. The `c2` command uses the file corresponding to additional pulses along the same direction. The `d` command uses the later pulse state represented by `d`. In each case, the script interprets the current-induced change by combining the inferred region tilts from the base state with a region-dependent activation map estimated from the change between the base and target spectral feature fields.

If the geometry-only effect of tilting the base state is what needs to be examined, the script can be run with the base and target made identical:

```bash
python tase2_geometry_simulation.py --base-case a --target-case a
```

This command does not simulate a pulse-induced transition toward a different measured state. Instead, it runs in geometry-only mode. The script loads only the base state, infers the valid mask, extracts the base feature set, infers regions, estimates region tilts from the base scan itself, warps the base cube by those inferred region tilts, and reports how different the geometry-warped state is from the original base state. This is useful for isolating how much of the spatial variation can be explained by local tilt and momentum-cut misregistration alone.

The default data location can be overridden. If the data are stored somewhere other than `../../data`, the command can be written as

```bash
python tase2_geometry_simulation.py --data-dir /absolute/path/to/data --target-case b
```

This tells the script to resolve the canonical case files from that directory instead of the default one. The behavior of the simulation remains the same; only the data source directory changes.

The output location can likewise be controlled explicitly. For example,

```bash
python tase2_geometry_simulation.py --target-case b --output-dir ../../outputs/custom_geometry_runs
```

This writes the dated run folder to the supplied directory instead of the default `../../outputs/geometry_simulation`. The script still creates a run-specific subdirectory named `geometry_simulation_<case>_YYYY_MM_DD` and writes the NetCDF file, JSON metrics, JSON region-parameter file, and optionally the `.npy` region products there.

The base and target files can be provided directly rather than through the case-name shorthand. This is useful if there are alternate corrected datasets, renamed files, or testing copies. The command has the form

```bash
python tase2_geometry_simulation.py \
  --base-file /absolute/path/to/a_convert_2_nosm.nc \
  --target-file /absolute/path/to/b_convert_2_nosm.nc
```

When `--base-file` and `--target-file` are supplied, they override `--base-case` and `--target-case`. The script resolves no default file names in that situation. The console output will print the exact paths it uses.

The energy-window parameters used for feature extraction can be changed if the corrected data use a slightly different effective Fermi level or if a broader or narrower near-
`E_F` window is desired. A representative command is

```bash
python tase2_geometry_simulation.py \
  --target-case b \
  --fermi-level-ev 0.0 \
  --ef-window-ev 0.05 \
  --wide-window-ev 0.20
```

This changes the spectral feature extraction stage only. The near-
`E_F` fraction, wide-window fraction, centroids, entropy, sharpness, and local contrast statistics will all be recomputed using those window definitions. This matters because the inferred region map and inferred current-activation map depend on the spectral features.

The number of inferred regions can be changed with

```bash
python tase2_geometry_simulation.py --target-case b --n-regions 8
```

This tells the script to partition the valid part of the sample into eight regions instead of the default six. The regions are inferred by clustering a subset of the standardized spectral feature fields extracted from the base scan. Increasing the number of regions allows the tilt and current-response parameters to vary more finely across the sample, while decreasing it forces a coarser, more global partition.

The automatically inferred valid device mask can be tuned through the cross-mask parameters. A command such as

```bash
python tase2_geometry_simulation.py \
  --target-case b \
  --cross-threshold-quantile 0.50 \
  --cross-row-fraction 0.20 \
  --cross-col-fraction 0.20 \
  --cross-background-quantile 0.12 \
  --cross-pad 2
```

changes the threshold used to identify the active cross-shaped device region from the integrated intensity maps. The threshold quantile controls how high the average normalized intensity must be before a pixel is considered active. The row and column fractions control the occupancy thresholds used to identify persistent rows and columns in the active device area. The background quantile removes low-intensity background. The padding parameter dilates the mask by the given number of iterations. The output of this command is still the same set of files, but the valid mask and therefore the inferred regions and activation map may change substantially.

The inferred tilt strengths can be scaled globally. For example,

```bash
python tase2_geometry_simulation.py \
  --target-case b \
  --tilt-scale-phi 1.5 \
  --tilt-scale-ev 0.50
```

multiplies the automatically inferred `phi` shifts by `1.5` and the automatically inferred energy shifts by `0.50`. This does not change the region map, only the magnitude of the region-wise spectral warping applied before the current-state mixing is performed. This is useful when the automatically inferred tilt amplitudes are too weak or too strong relative to what is seen in the experiment.

The current-response strength can also be scaled globally:

```bash
python tase2_geometry_simulation.py --target-case b --current-gain 1.25
```

The `current-gain` parameter multiplies the inferred activation map before it is used in the region-wise mixing step. Increasing it makes the simulated state move more aggressively from the warped base toward the warped target. Decreasing it makes the simulated state remain closer to the warped base state. This is useful when the geometry model seems reasonable but the simulated magnitude of the pulse response is too weak or too strong.

The inferred region map can be smoothed by applying majority voting over neighboring pixels. The number of smoothing passes is controlled by

```bash
python tase2_geometry_simulation.py --target-case b --smooth-region-iterations 3
```

This reduces isolated region assignments and encourages more spatially coherent domains. The main output remains the same, but the region map becomes less noisy and the region boundaries become smoother.

A previously saved region map can be reused directly rather than re-inferred. The command is

```bash
python tase2_geometry_simulation.py \
  --target-case c \
  --region-map ../../outputs/geometry_simulation/geometry_simulation_b_YYYY_MM_DD/region_map_b_YYYY_MM_DD.npy
```

This tells the script to skip region inference and instead use the provided integer region map. The map must have the same `(x, y)` shape as the current dataset. This is one of the most important workflow commands, because it allows the same spatial partition to be reused across multiple target states or after manual editing of the region definitions.

A region map can also be reused together with saving the new run’s region products:

```bash
python tase2_geometry_simulation.py \
  --target-case d \
  --region-map ../../outputs/geometry_simulation/geometry_simulation_b_YYYY_MM_DD/region_map_b_YYYY_MM_DD.npy \
  --save-region-products
```

This is useful when using a hand-refined or previously validated region map to test a different pulse state while still saving the updated activation map and output files for that new target case.

A JSON file can be used to override the automatically inferred per-region parameters. The command is

```bash
python tase2_geometry_simulation.py \
  --target-case b \
  --region-params-json my_region_params.json
```

The JSON file is expected to contain per-region entries with `region_id`, and optionally `phi_shift_bins`, `energy_shift_bins`, `current_mix`, and `name`. This allows the region partition to stay fixed while the tilt or mixing parameters are edited manually between runs. This is especially useful for dynamic fitting against the experiment, where the user wants to increase or decrease the tilt strength in specific regions or alter how strongly those regions respond to the pulse.

A representative override file looks like this:

```json
{
  "regions": [
    {
      "region_id": 0,
      "phi_shift_bins": 1.5,
      "energy_shift_bins": -0.2,
      "current_mix": 0.8,
      "name": "bottom_left_domain"
    },
    {
      "region_id": 1,
      "phi_shift_bins": -0.7,
      "energy_shift_bins": 0.1,
      "current_mix": 0.3,
      "name": "upper_boundary_region"
    }
  ]
}
```

When this file is supplied, only the specified fields are replaced. Regions not mentioned remain at their automatically inferred values.

The script can also be run with both a fixed region map and a fixed parameter-override file. The command is

```bash
python tase2_geometry_simulation.py \
  --target-case b \
  --region-map my_region_map.npy \
  --region-params-json my_region_params.json \
  --save-region-products
```

This is the most controlled mode of operation. The script skips region inference, loads the user-defined region map, loads the automatically inferred parameters, applies the user’s overrides, performs the region-wise geometry warping and current-response mixing, and writes the final simulated results. This is the mode that best supports interactive refinement of the model against experimental data.

## What the script writes

Every run creates a dated directory of the form

```text
../../outputs/geometry_simulation/geometry_simulation_<target_case>_YYYY_MM_DD/
```

unless a different `--output-dir` is supplied. Within that run directory, the script writes a NetCDF file named

```text
simulated_<target_case>_YYYY_MM_DD.nc
```

This NetCDF file contains the simulated intensity cube and several 2D auxiliary products. The primary variable is `simulated_intensity`, which has the same dimensions and coordinates as the original data cube. The file also contains `region_map`, `valid_mask`, and `activation_map` as spatial maps on the `(x, y)` grid. The dataset attributes include run metadata such as the base file, target file, number of regions, and error metrics.

The script also writes a metrics JSON file named

```text
metrics_<target_case>_YYYY_MM_DD.json
```

This file stores summary run information, including the resolved base and target file paths, number of valid pixels, number of excluded pixels, number of inferred regions, and the root-mean-square error between the simulated and target near-
`E_F` fraction map and total intensity map when a distinct target file is present.

A region-parameter JSON file is always written:

```text
region_params_<target_case>_YYYY_MM_DD.json
```

This contains the final per-region `phi_shift_bins`, `energy_shift_bins`, `current_mix`, and region names used in the run. It serves both as a record of the run and as a template for later manual editing through `--region-params-json`.

A summary figure is also written. By default it is first rendered in the top-level output directory with the name

```text
geometry_simulation_summary_<target_case>_YYYY_MM_DD.png
```

and then copied into the run directory. This image shows the average normalized total intensity map, the inferred or loaded region map, the activation map, the base near-
`E_F` fraction, the simulated near-
`E_F` fraction, and the simulated-minus-target or simulated-minus-base difference map.

If `--save-region-products` is supplied, the following additional files are written to the run directory:

```text
region_map_<target_case>_YYYY_MM_DD.npy
valid_mask_<target_case>_YYYY_MM_DD.npy
activation_map_<target_case>_YYYY_MM_DD.npy
```

These are useful for region reuse, manual editing, and downstream analysis.

## What the console output means

The script prints timestamped progress lines throughout the run. Typical messages include attempts to open the dataset with different `xarray` backends, confirmation of which backend succeeded, when the base and target cubes are materialized as NumPy arrays, the number of valid and excluded pixels in the inferred device mask, when base and target features are extracted, whether the region map is being inferred or loaded, when region parameters are inferred or overridden, when the base and target cubes are warped, and a line for each region giving its region id, the number of pixels in that region, and the values of `phi_shift_bins` and `energy_shift_bins` that are being applied.

At the end of the run, the script prints the resolved base file path, the resolved target file path, the final NetCDF output path, the summary figure path, and the region parameter values for every region. This final block is the quickest way to inspect what the run actually did.

## Practical command sequence for iterative fitting

A useful first run is

```bash
python tase2_geometry_simulation.py --target-case b --save-region-products
```

This produces an automatically inferred region map and parameter file.

The next step is to inspect the saved figure, region map, and parameter JSON file, then rerun using the saved region map and a manually edited parameter file:

```bash
python tase2_geometry_simulation.py \
  --target-case b \
  --region-map ../../outputs/geometry_simulation/geometry_simulation_b_YYYY_MM_DD/region_map_b_YYYY_MM_DD.npy \
  --region-params-json edited_region_params.json \
  --save-region-products
```

If the same region partition is meant to be tested on a different pulse state, the command becomes

```bash
python tase2_geometry_simulation.py \
  --target-case c \
  --region-map ../../outputs/geometry_simulation/geometry_simulation_b_YYYY_MM_DD/region_map_b_YYYY_MM_DD.npy \
  --region-params-json edited_region_params.json \
  --save-region-products
```

This is the recommended workflow for examining whether the same region-based geometric interpretation carries across multiple pulse states.
