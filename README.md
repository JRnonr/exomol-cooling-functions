### ExoMol Cooling Functions Project Overview

This repository computes isotopologue cooling functions based on the ExoMol database and provides multi-process/multi-thread acceleration along with natural-abundance weighted plotting tools.


## Directory Structure
- `cooling/`: per-molecule folders, computed `.cf` results, and the natural-abundance outputs and plots for each molecule (`natural_abundance.cf` and `natural_abundance.png`)
- `pyexocross_thread.py`: an optimized, streamlined version based on the original `pyexocross.py` with several fixes
- `pyexocross_process_local.py`: `ProcessPoolExecutor` version for local machine; typically runs `light.txt`
- `pyexocross_process_exoweb.py`: `ProcessPoolExecutor` version for ExoWeb server; typically runs `heavy.txt`
- `pyexocross_process_dias.py`: `ProcessPoolExecutor` version for DIAS server; typically runs `heavy.txt`
- `light.txt` / `heavy.txt`: task lists (each line is an `Isotopologue__Dataset`)
- `cooling_status_updated.csv`: latest status tracking for cooling function computations
- `natural_abundance_plot.py`: natural-abundance weighting over multiple isotopologue `.cf` files for the same molecule, producing `natural_abundance.cf` and a PNG plot
- `isotopic_abundance.json`: natural abundance weights
