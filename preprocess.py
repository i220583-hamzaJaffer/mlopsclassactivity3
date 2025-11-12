import pandas as pd
from sklearn.datasets import load_iris
import os # Import the os module

# --- Data Loading and Preparation (Original Code) ---
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# --- FIX: Create the output directory before saving the file ---
output_dir = "data"
if not os.path.exists(output_dir):
    # The exist_ok=True argument prevents an error if the directory already exists
    os.makedirs(output_dir, exist_ok=True) 

# --- Save the file ---
data.to_csv(f"{output_dir}/preprocessed.csv", index=False)
print(f"✅ Data preprocessed and saved to {output_dir}/preprocessed.csv")
