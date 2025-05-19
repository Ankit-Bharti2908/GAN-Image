#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
from IPython.display import display, HTML, IFrame

# Make sure we have streamlit installed
try:
    import streamlit
except ImportError:
    print("Installing Streamlit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])

# Create the necessary directories if they don't exist
os.makedirs("./outputs", exist_ok=True)

# Save the UI files
def save_ui_files():
    """
    Save the UI files to the current directory.
    Returns the paths to the created files.
    """
    # Create the UI files content based on the artifacts
    dualstylegan_functions_content = """
# This file contains all the functions for DualStyleGAN
# It was generated automatically - see the full content in the notebook

import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import torchvision
from torch.nn import functional as F
import tempfile
import time
import sys
import dlib

# Add the DualStyleGAN directory to the path
sys.path.append("./DualStyleGAN")

# Import DualStyleGAN modules
from model.dualstylegan import DualStyleGAN
from model.sampler.icp import ICPTrainer
from model.encoder.psp import pSp
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants
MODEL_DIR = './DualStyleGAN/checkpoint'
DATA_DIR = './DualStyleGAN/data'
OUTPUT_DIR = './outputs'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Available style types
style_types = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']

# Create transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# See full implementation in the notebook
# The rest of the functions are implemented there
"""

    dualstylegan_ui_content = """
# This file contains the Streamlit UI components
# It was generated automatically - see the full content in the notebook

import streamlit as st
import os
import tempfile
import time
import zipfile
import shutil
import base64
from PIL import Image

from dualstylegan_functions import (
    load_models, get_dlib_predictor, style_types,
    get_style_names, process_image, create_style_weights_grid,
    create_style_blend, create_blend_sequence, sample_random_styles,
    process_image_custom_style, create_custom_style_interpolation_grid,
    find_optimal_parameters, tensor_to_pil, get_image_download_link,
    check_checkpoints, MODEL_DIR, DATA_DIR, OUTPUT_DIR
)

# See full implementation in the notebook
# The rest of the UI components are implemented there
"""

    launcher_content = """
import streamlit as st
from dualstylegan_ui import run_app

if __name__ == "__main__":
    run_app()
"""

    # Write files to disk
    functions_path = "dualstylegan_functions.py"
    ui_path = "dualstylegan_ui.py"
    launcher_path = "streamlit_app.py"
    
    # Check if files already exist, and don't overwrite
    files_to_write = [
        (functions_path, dualstylegan_functions_content),
        (ui_path, dualstylegan_ui_content),
        (launcher_path, launcher_content)
    ]
    
    created_files = []
    for file_path, content in files_to_write:
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Created {file_path}")
        else:
            print(f"{file_path} already exists")
        created_files.append(file_path)
    
    return created_files

# Save UI files
ui_files = save_ui_files()

# Function to launch the Streamlit app
def launch_streamlit_app():
    """
    Launch the Streamlit app in a Kaggle notebook.
    """
    from pyngrok import ngrok
    
    # Check for ngrok installation
    try:
        import pyngrok
    except ImportError:
        print("Installing pyngrok...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        from pyngrok import ngrok
    
    # Set up ngrok - use the authtoken if provided
    ngrok_token = os.environ.get("NGROK_TOKEN", "")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
    
    # Kill any existing processes on port 8501
    try:
        subprocess.run(['kill', '-9', '$(lsof -t -i:8501)'], shell=True, check=False)
    except:
        pass
    
    # Start the Streamlit app in the background
    print("Starting the Streamlit app...")
    cmd = f"streamlit run streamlit_app.py --server.port 8501 &"
    subprocess.Popen(cmd, shell=True)
    
    # Give the server a moment to start
    time.sleep(5)
    
    # Create a ngrok tunnel
    ngrok_tunnel = ngrok.connect(8501)
    ngrok_url = ngrok_tunnel.public_url
    
    print(f"Streamlit app running at: {ngrok_url}")
    
    # Display the app in an iframe
    display(HTML(f'<a href="{ngrok_url}" target="_blank">Click here to open DualStyleGAN UI in a new tab</a>'))
    display(IFrame(ngrok_url, width=800, height=600))

# Launch the Streamlit app from notebook if desired
print("\nDualStyleGAN UI files created. To launch the UI, run the following cell:")
print("from launch_dualstylegan_ui import launch_streamlit_app")
print("launch_streamlit_app()")
