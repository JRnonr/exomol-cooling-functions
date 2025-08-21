import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_isotope(isotope_str):
    match = re.match(r"^(\d+)([A-Za-z]+)$", isotope_str)
    if not match:
        raise ValueError(f"Unrecognized isotope format: {isotope_str}")
    mass, element = match.groups()
    return element, isotope_str

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
    print(f"🔍 Searching .cf files in: {cf_dir}")
    # cf_files = [f for f in os.listdir(cf_dir) if f.endswith(".cf") and not f.startswith("natural_abundance")]
    cf_files = []
    for root, _, files in os.walk(cf_dir):
        for file in files:
            if file.endswith(".cf") and not file.startswith("natural_abundance"):
                cf_files.append(os.path.join(root, file))

    if not cf_files:
        print("[❌] No valid .cf files found.")
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
                element, iso_key = parse_isotope(iso)
                weight *= abundance_json[element][iso_key]

            T, val = read_cf(os.path.join(cf_dir, file))
            if T is None:
                continue
            if all_temperatures is None:
                all_temperatures = T
            val_weighted = val * weight
            cooling_data.append(val_weighted)
            weights.append(weight)
            label_data.append((iso_name, T, val_weighted, weight))
            print(f"[✔️] {iso_name}: weight = {weight:.5f}")
        except Exception as e:
            print(f"Skipped {file}: {e}")

    if cooling_data:
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


# === Main entry point ===
if __name__ == "__main__":
    cf_dir = "cooling/HCl"
    abundance_path = "isotopic_abundance.json"
    output_cf_name = "natural_abundance.cf"
    output_img = os.path.join(cf_dir, "natural_abundance.png")

    with open(abundance_path) as f:
        abundance_json = json.load(f)

    # Compute natural-abundance-weighted cooling function
    natural_cf_path, _ = weighted_cooling_function(cf_dir, abundance_json, output_path=output_cf_name)

    # Plot the natural abundance curve
    if natural_cf_path:
        plot_natural_only(natural_cf_path, output_img)
