import streamlit as st
import numpy as np
import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
import os
import subprocess

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Command '{command}' failed with error: {result.stderr}")
        raise RuntimeError(f"Command '{command}' failed with error: {result.stderr}")



st.set_page_config(page_title="BioMiMiC", page_icon=":molecule:")

# Add the image to the left of the text
col1, _, col2 = st.columns([0.65, 0.05, 0.25])
with col1:
    st.markdown(
        """
        <h1 style='text-align: right;'>BioMiMiC: Bioactive Molecule Discovery</h1>
        <div style='text-align: right;'>
            BioMiMiC is an AI solution for researchers and industries aiming to discover sustainable, bioactive molecules. By using existing molecules, users can easily train a predictive model and then search on un-labelled natural molecule databases.
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.image("img/group-1.png", width=150)

# Step 1: Fine-tuning a predictive model
st.header("Step 1: Fine-tune a Predictive Model")
st.write("Enter a molecular function and click the button to start fine-tuning.")

# User input for molecular function
molecules_functions = ["Antioxydant", 'Antibiotique', 'Antifongique', 'Analgésique']
# mol_function = st.text_input("Molecular Function")
mol_function = st.selectbox("Molecular Function", molecules_functions)


if st.button("Fine-tune Model"):
    st.write(f"Fine-tuning model for '{mol_function}' function...")
    # Perform fine-tuning process here
    st.success("Model fine-tuning completed!")

# Step 2: Screen Databases
st.header("Step 2: Screen Databases")
st.write("Select a pre-trained model and click the button to screen the Coconut database.")

# Dropdown for pre-trained models
pre_trained_models = ["Model A", "Model B", "Model C"]
selected_model = st.selectbox("Select a Pre-trained Model", pre_trained_models)

if st.button("Screen Coconut Database"):
    st.write(f"Screening Coconut database using '{selected_model}'...")
    # Perform screening process here

    # Check if the Conda environment is already activated
    env_name = 'hugging_face'
    # if not os.environ.get('CONDA_DEFAULT_ENV', '') == env_name:
        # Activate the Conda environment
    # run_command(['conda', 'init', 'zsh'])
    result = subprocess.run(['conda', 'init', 'zsh'], capture_output=True, text=True)
    if result.returncode == 0:
        st.success("Screening completed!")
    else:
        st.error(f"Screening failed with error: {result.stderr}")
 
        # run_command(['conda', 'activate', env_name])

    result = subprocess.run(['conda', 'init', 'bash', '&&', 'conda', 'activate', 'hugging_face', '&&', 'python', '../SMILES-BERT/predict.py../SMILES-BERT/predict.py'], capture_output=True, text=True)
    # os.system('python ../SMILES-BERT/predict.py')
    if result.returncode == 0:
        st.success("Screening completed!")
    else:
        st.error(f"Screening failed with error: {result.stderr}")
    # st.success("Screening completed!")

# Step 3: Download Results
st.header("Step 3: Download Results")
st.write("Click the button to download the screening results.")

if st.button("Download Results"):
    # Generate a sample result file
    result_data = pd.DataFrame({
        "Molecule": ["Molecule A", "Molecule B", "Molecule C"],
        "Bioactivity Score": [0.8, 0.6, 0.9]
    })
    
    # Save the result file
    result_file = "screening_results.csv"
    result_data.to_csv(result_file, index=False)
    
    st.download_button(
        label="Download Results",
        data=result_data.to_csv(index=False),
        file_name=result_file,
        mime="text/csv",
    )
    st.success("Results downloaded successfully!")

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data
