import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re


def parse_isotope(isotope_str):
    # Support formats like 1H, 16O, 1H2 (count suffix), allow element letters only
    match = re.match(r"^(\d+)([A-Za-z]+)(\d*)$", isotope_str)
    if not match:
        raise ValueError(f"Unrecognized isotope format: {isotope_str}")
    mass, element, count_str = match.groups()
    count = int(count_str) if count_str else 1
    iso_key = f"{mass}{element}"
    return element, iso_key, count


def read_cf(filepath):
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return None, None


def weighted_cooling_function(cf_dir, abundance_json, output_path="natural_abundance.cf"):
    print(f"Searching .cf files in: {cf_dir}")
    # cf_files = [f for f in os.listdir(cf_dir) if f.endswith(".cf") and not f.startswith("natural_abundance")]
    cf_files = []
    for root, _, files in os.walk(cf_dir):
        for file in files:
            if file.endswith(".cf") and not file.startswith("natural_abundance"):
                cf_files.append(os.path.join(root, file))

    if not cf_files:
        print("No valid .cf files found.")
        return None, []

    cooling_data = []
    weights = []
    all_temperatures = None
    label_data = []

    for file in cf_files:
        try:
            basename = os.path.basename(file)
            iso_name = basename.replace(".cf", "").split("__")[0]
            isotopes = iso_name.split("-")
            weight = 1.0
            for iso in isotopes:
                # Trim at the first non-alphanumeric char (e.g., 1H_p -> 1H)
                m = re.match(r'^[0-9A-Za-z]+', iso)
                if not m:
                    raise ValueError(f"Unrecognized isotope token: {iso}")
                iso_clean = m.group(0)
                element, iso_key, count = parse_isotope(iso_clean)
                weight *= abundance_json[element][iso_key] ** count

            # file is already a full path collected from os.walk
            T, val = read_cf(file)
            if T is None:
                continue

            # Establish common temperature grid from the first valid file
            if all_temperatures is None:
                all_temperatures = T

            # Interpolate onto the common grid if needed; outside range -> NaN -> treated as 0
            if not np.allclose(T, all_temperatures):
                try:
                    val = np.interp(all_temperatures, T, val, left=np.nan, right=np.nan)
                except Exception as e:
                    print(f"Interpolation failed for {file}: {e}")
                    continue

            val_weighted = np.nan_to_num(val * weight, nan=0.0)
            cooling_data.append(val_weighted)
            weights.append(weight)
            label_data.append((iso_name, all_temperatures, val_weighted, weight))
            print(f"{iso_name}: weight = {weight:.5f}")
        except KeyError as e:
            print(f"Skipped {file}: {e}")
        except Exception as e:
            print(f"Skipped {file}: {e}")

    if cooling_data and all_temperatures is not None:
        total_cooling = np.sum(cooling_data, axis=0)
        result = np.column_stack((all_temperatures, total_cooling))
        output_full_path = os.path.join(cf_dir, output_path)
        np.savetxt(output_full_path, result, fmt="%.6e")
        print(f"Weighted cooling function saved to: {output_full_path}")
        return output_full_path, label_data
    else:
        print("No valid cooling functions processed.")
        return None, []


def plot_natural_only(natural_cf_path, output_img_path):
    """
    Figure: plot only the natural-abundance-weighted cooling function.
    """
    T_nat, val_nat = read_cf(natural_cf_path)
    if T_nat is None or val_nat is None:
        print("Failed to read natural abundance cooling function file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(T_nat, val_nat, label="natural_abundance", linewidth=2.5, color="green")
    plt.yscale("log")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Cooling Function (erg s$^{-1}$ cm$^3$)")
    plt.title("Natural-Abundance-Weighted Cooling Function")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300)
    plt.close()
    print(f"Natural abundance plot saved to: {output_img_path}")


def generate_for_all_molecules(root_dir, abundance_json):
    """
    Iterate all molecule folders under root_dir, and for those with >= 2 .cf files,
    generate natural-abundance-weighted cooling function and plot it into the same folder.
    """
    if not os.path.isdir(root_dir):
        print(f"Root directory does not exist: {root_dir}")
        return

    molecule_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    processed = 0
    skipped = 0

    for mol_dir in molecule_dirs:
        # Recursively count .cf files under the molecule folder (excluding previously generated natural cf)
        cf_files = []
        for r, _, files in os.walk(mol_dir):
            for f in files:
                if f.endswith('.cf') and not f.startswith('natural_abundance'):
                    cf_files.append(os.path.join(r, f))

        if len(cf_files) < 2:
            print(f"Skip {mol_dir}: fewer than 2 isotopologue .cf files")
            skipped += 1
            continue

        print(f"Processing {mol_dir} with {len(cf_files)} .cf files ...")
        natural_cf_name = "natural_abundance.cf"
        output_img = os.path.join(mol_dir, "natural_abundance.png")

        natural_cf_path, _ = weighted_cooling_function(mol_dir, abundance_json, output_path=natural_cf_name)
        if natural_cf_path:
            plot_natural_only(natural_cf_path, output_img)
            processed += 1
        else:
            skipped += 1

    print(f"Done. Processed: {processed}, Skipped: {skipped}")


# === Main entry point ===
if __name__ == "__main__":
    # Root folder that contains molecule subfolders
    root_dir = "cooling"
    # Absolute path to isotopic abundance JSON
    abundance_path = "isotopic_abundance.json"

    with open(abundance_path) as f:
        abundance_json = json.load(f)

    # One-click generation for all molecules
    generate_for_all_molecules(root_dir, abundance_json)
