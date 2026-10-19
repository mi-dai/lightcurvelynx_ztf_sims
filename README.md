# lightcurvelynx_ztf_sim

Simulations of the Zwicky Transient Facility (ZTF) Type Ia supernova survey using [LightCurveLynx](https://github.com/lincc-frameworks/lightcurvelynx).

The simulations use the real ZTF observing conditions: the ZTF public metadata database is combined with the per-CCD-quadrant observing log from the ZTF SN Ia DR2 release. SN Ia parameter distributions, the DR2 selection function and the light-curve quality cuts are calibrated against the DR2 sample. The simulated light curves are fit with SALT3 (via `sncosmo`) and compared with the DR2 data. The notebooks also produce the figures for [Dai et al. (2026)](https://arxiv.org/abs/2604.07134).

## Repository layout

| Path | Contents |
| --- | --- |
| `ztf_snia_sim_params.py` | Shared simulation parameters (`SIM_PARAMS`): cosmology, SALT standardization (α, β, M, σ_M), and the redshift / x1 / c ranges |
| `utils/gen_ztf.py` | Data loaders, matching a SN to its quadrant-level observing log, and `gen_single_ztf_sn_lc` for simulating an individual DR2 SN |
| `utils/analysis_utils.py` | Light-curve quality cuts, spectroscopic selection function, SN–host separation, and sky noise derived from limiting magnitudes |
| `utils/lcfit.py` | `fit_single_lc`: SALT3 fit of a single light curve |
| `utils/plotting_utils.py` | Plots that compare simulations with data (flux and flux-error distributions, SNR, sky coverage) |
| `utils/ztfdata_utils.py` | `parse_all_lightcurves`: reads the DR2 light curves into a `NestedFrame` |
| `data/` | Input data products (combined observing log, selection functions, pzflow models) |
| `ztfsniadr2/` | ZTF SN Ia DR2 release (tables, light curves, spectra) |
| `results/` | Simulation outputs and SALT3 fit results |
| `figs/` | Figures embedded in the notebooks |
| `paper_figs/` | Paper figures |
| `ancillary/` | Exploratory and archived notebooks |

## Workflow

Run the notebooks from the repository root, because they use relative paths such as `data/...` and imports such as `from utils...`.

1. **Preprocessing**
   - `ztf_observing_log.ipynb` combines the ZTF metadata DB with the DR2 observing log to build `data/ztf_observing_log_combined_w_metadata.parquet`.
2. **Calibrating the population model**
   - `ZTF_snia_luminosity.ipynb`: check luminosity function.
   - `ZTF_snia_selection_func.ipynb` simulates SNe with flat x1/c distributions to derive the x1 and c selection functions (`data/ztf_selection_func_{x1,c}.txt`) and the absolute-magnitude offset caused by selection effects.
   - `train_pzflow.ipynb` trains a [pzflow](https://github.com/jfcrenshaw/pzflow) model of x1, c and host mass, then removes the selection function to get the distributions before selection (`data/ztfsniadr2_host_sn_*pzflow.pkl`).
3. **Simulations**
   - `ZTF_snia-gen-sims-and-fitlc.ipynb` is the main survey simulation. It uses realistic cadence, rates and parameter distributions, applies the DR2 selection and quality cuts, and fits SALT3. Outputs go to `results/`.
   - `ZTF_snia-gen-single-sim.ipynb` and `ZTF_snia-gen-ztf-match-sim.ipynb` simulate real DR2 SNe (one SN, or all of them) on their exact CCD-quadrant observing log and compare the simulations with the observed light curves.
4. **Paper figures**
   - `ZTF_snia-paper-plots.ipynb` makes the figures and tables for Dai et al. (2026) and saves them to `paper_figs/`.

## Data requirements

Git does not track `ztfsniadr2/`, `results/`, `paper_figs/` or some of the files in `data/` (see `.gitignore`). You need to get these yourself:

- **ZTF SN Ia DR2**: put the release in `ztfsniadr2/`, with its `tables/` and `lightcurves/` folders.
- **ZTF metadata DB**: put it at `data/ztf_metadata_latest.db`. The exposure metadata comes from the [ZTF public data release](https://irsa.ipac.caltech.edu/data/ZTF/docs/releases/ztf_release_notes_latest) at IRSA.
- **DR2 light curves as parquet**: generate `data/ztfsniadr2.parquet` with `utils.ztfdata_utils.parse_all_lightcurves("ztfsniadr2")`.
- **SFD dust maps**: set the `SFD_DIR` environment variable to the SFD data directory. `utils/lcfit.py` needs it for Milky Way extinction.

## References

- Rigault et al. (2025), *ZTF SN Ia DR2: Overview*, [A&A 694, A1](https://ui.adsabs.harvard.edu/abs/2025A%26A...694A...1R/abstract)
- Dai et al. (2026), *LightCurveLynx: Forward Modeling of Time-Domain Surveys with Application to ZTF SN Ia DR2*, [arXiv:2604.07134](https://arxiv.org/abs/2604.07134)
- [LightCurveLynx](https://github.com/lincc-frameworks/lightcurvelynx)

## License

MIT. See [LICENSE](LICENSE).
