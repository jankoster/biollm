import os
import sys
import argparse

# %% [markdown]
# Set variables
project_folder = "raymond"
filename_root_ref = "Fskin_obj_2_4_1_webatlas"
ref_meta_colname = "annotation_fine"
filename_root_target_manual = ''

sample_id = "sanger_id"
sample_id_target = "HTO_maxID"
predictions_meta_name = "cell_type_predictions"

# Define categories of interest from ref data (fibroblast example)
categories_of_interest = [
    "CCL19+ fibroblast",
    "FRZB+ early fibroblast",
    "HOXC5+ early fibroblast",
    "Myofibroblasts",
    "PEAR1+ fibroblast",
    "WNT2+ fibroblast",
]

foldername_input_base = "/home/lieke_unix/input/"
foldername_output_base = "/home/lieke_unix/output/"

foldername_input = os.path.join(foldername_input_base, project_folder)
foldername_output = os.path.join(foldername_output_base, project_folder)
filename_ref= os.path.join(foldername_input, filename_root_ref + ".h5ad")


# %% 
# Setting arguments
if __name__ == "__main__":  
    if len(sys.argv) > 1:  # means you're running via command line with args
        parser = argparse.ArgumentParser(description="Run cell type annotation and log output automatically.")
        parser.add_argument("filename_root_target", type=str, help="Dataset name for this run, used in logs and filenames.")
        parser.add_argument("--tee", action="store_true", help="Optional. If set, print to terminal as well as log file")
        args = parser.parse_args()

        filename_root_target = args.filename_root_target
        use_tee = args.tee
    else:
        # fallback for interactive runs
        filename_root_target = filename_root_target_manual   
        tee_enabled = True    

filename_target = os.path.join(foldername_input, filename_root_target + ".h5ad")
filename_out_base = os.path.join(foldername_output, filename_root_target)
filename_out_predictions = filename_out_base + "_celltype_preds.csv"



# -------------------------------
# Prepare log file with timestamp
# -------------------------------
os.makedirs(foldername_output, exist_ok=True)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(foldername_output, f"output_{filename_root_target}_{timestamp}.log")

# -------------------------------
# Redirect stdout/stderr
# -------------------------------
if use_tee:
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    sys.stdout = Tee(sys.stdout, open(log_file_path, "w"))
    sys.stderr = Tee(sys.stderr, open(log_file_path, "w"))
else:
    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout


print(f"Finetuning the scGPT model will be done with: {filename_ref}")
print(f"Starting processing for target file: {filename_target}")
print(f"Logging to: {log_file_path}")