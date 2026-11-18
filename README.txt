================================================================================
Reproduction package

  What the annual energy balance hides: the coordination value of digital
  platforms for positive energy districts
================================================================================


CITATION
--------------------------------------------------------------------------------
Abate, A. G., Nielsen, P. S., Liu, X., O'Regan, B., Cebrat, G., Karadag, C.,
Zaccaria, V., Liu, Y., Silion, M., Vadiee, A., Jespersen, J., Bueker, M. S.,
Raboaca, M. S., Renuke, A., Erdoer Tuerk, B., Stewart, D.
What the annual energy balance hides: the coordination value of digital
platforms for positive energy districts. Applied Energy.

Add the DOI and volume once assigned.


CONTENTS
--------------------------------------------------------------------------------
  model.py          Optimization model and solution routines: the mixed-integer
                    community problem, its convex relaxation, the price-taking
                    peer-to-peer equilibrium, and the joint solve returning the
                    dual prices.
  analysis.py       Experiments and table generation.
  plottings.py      Figure generation.
  config.py         Model constants.
  data.py           Hourly profile construction.
  data/             Inputs and cached results.
  figures/          Figures as published.
  requirements.txt  Dependencies.
  LICENSE           License terms.

config.py and data.py are imported by all three entry points and are kept as
separate modules.


USAGE
--------------------------------------------------------------------------------
  pip install -r requirements.txt

  python analysis.py tables        writes tables/*.tex
  python plottings.py figures      writes figures/*.pdf and *.png

Both read data/results only. Neither requires a solver. Each completes in under
a minute.

  python analysis.py experiments   re-solves everything, requires Gurobi

One thread, seed 42, optimality gap 1e-4. Approximately one hour. Rewrites
data/results.

  python plottings.py fetch-basemap

Caches the Natural Earth shaded-relief raster used by the site map, about
700 MB, under ~/.local/share/cartopy/rasters.


REGENERATED FROM INPUTS, OR FROM CACHED OUTPUTS
--------------------------------------------------------------------------------
Regenerated from inputs (requires Gurobi)

  All experiment results in data/results. The hourly profiles are constructed
  deterministically by data.py from the constants in config.py. No measured
  series is used and none is required.

Reproduced from cached outputs (no solver)

  All tables and all figures. They are computed from the JSON files in
  data/results.

Two files in data/results were produced on a compute cluster rather than by
the experiment command, because of their running time.

  thermal_mass.json    pre-heating sensitivity behind Figure 10
  best_response.json   unilateral withholding check, Section 5.6


VERIFICATION
--------------------------------------------------------------------------------
On the cluster environment named in the article's data availability statement,
with gurobipy not installed, both commands completed and all 13 figures were
produced, including the site map at 1.70 MB and 312 dpi. All 11 figures used in
the article matched the published versions on their text content.

On a local workstation the same commands produced all figures except the site
map, which failed while reading the cached Natural Earth raster
(TIFFReadEncodedStrip). The failure is in the local raster cache, not in this
package. The site map shipped in figures/ is the cluster-rendered version.

analysis.py tables writes 22 LaTeX fragments. The article uses 12 of them.


ENVIRONMENT
--------------------------------------------------------------------------------
Results in the article come from

  Python 3.12.11, Gurobi 13.0.0, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2,
  Cartopy 0.26.0, Rasterio 1.5.2

Gurobi is commercial and requires a license, free for academic use. It is needed
only by analysis.py experiments.

The optional convex check of the perfect-information stages uses CVXPY with
Clarabel and reproduces the Gurobi values to within the 1e-4 optimality
tolerance.


FIGURE 12
--------------------------------------------------------------------------------
The right panel plots four stage contributions against one denominator, the
DE0 to SP gap, and they sum to one. Table 8 of the article reports the two
perfect-information contributions against the IND to SP distance, which is why
those entries are larger. The axis label states the denominator.


DATA PROVENANCE
--------------------------------------------------------------------------------
All hourly series are constructed, not measured. Construction is deterministic
and seeded.

Parameters are calibrated to public sources.

  Day-ahead price levels and shapes, zonal emission factors
      ENTSO-E Transparency Platform, 2023-24
  Volumetric use-of-system tariffs
      European electricity price statistics, 2023-24
  Country photovoltaic specific yields
      PVGIS climate data, typical meteorological year
  Feed-in ratios
      Stylized national values, 2023-24, scenario assumption
  Net-export hosting limits
      Scenario assumption, swept in the sensitivity analysis

  data/countries_110m.geojson, data/countries_50m.geojson
      Natural Earth administrative boundaries, public domain
  Shaded-relief raster fetched by plottings.py fetch-basemap
      Natural Earth I with shaded relief, water and drainage, public domain

Validation against measured pilot data is identified in the article as future
work and is not part of this package.


REPEATABILITY
--------------------------------------------------------------------------------
Two complete pipeline passes were compared value by value and agree exactly on
38,702 quantities. Core results were reproduced on a second Gurobi build,
13.0.2 against 13.0.0, with a worst absolute deviation below 1e-8 across 54
compared quantities spanning all six configurations.
