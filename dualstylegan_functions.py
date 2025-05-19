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

# Helper function to get style names for a style type
def get_style_names(style_type):
    exstyles_path = os.path.join(MODEL_DIR, style_type, 'refined_exstyle_code.npy')
    if not os.path.exists(exstyles_path):
        exstyles_path = os.path.join(MODEL_DIR, style_type, 'exstyle_code.npy')
    
    if os.path.exists(exstyles_path):
        exstyles = np.load(exstyles_path, allow_pickle='TRUE').item()
        return list(exstyles.keys())
    return []

# Function to run face alignment
def get_dlib_predictor():
    modelname = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        print("Downloading face landmark predictor. This may take a moment...")
        import wget
        import bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        with bz2.BZ2File(modelname+'.bz2') as zipfile:
            data = zipfile.read()
            with open(modelname, 'wb') as f:
                f.write(data)
    
    return dlib.shape_predictor(modelname)

def run_alignment(image_path, predictor):
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    return aligned_image

# Load models
def load_models(style_type):
    print(f"Loading {style_type} models...")
    # Load DualStyleGAN
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    generator.eval()
    ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'generator.pt'), 
                     map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)
    
    # Load encoder
    model_path = os.path.join(MODEL_DIR, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    from argparse import Namespace
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder = encoder.to(device)
    
    # Load extrinsic style code
    exstyles_path = os.path.join(MODEL_DIR, style_type, 'refined_exstyle_code.npy')
    if not os.path.exists(exstyles_path):
        exstyles_path = os.path.join(MODEL_DIR, style_type, 'exstyle_code.npy')
    
    exstyles = np.load(exstyles_path, allow_pickle='TRUE').item()
    
    # Load sampler network
    icptc = ICPTrainer(np.empty([0,512*11]), 128)
    icpts = ICPTrainer(np.empty([0,512*7]), 128)
    ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'sampler.pt'), 
                     map_location=lambda storage, loc: storage)
    icptc.icp.netT.load_state_dict(ckpt['color'])
    icpts.icp.netT.load_state_dict(ckpt['structure'])
    icptc.icp.netT = icptc.icp.netT.to(device)
    icpts.icp.netT = icpts.icp.netT.to(device)
    
    print("Models successfully loaded!")
    return generator, encoder, exstyles, icptc, icpts

# Function to process image and generate stylized results
def process_image(image_path, generator, encoder, exstyles, style_name, 
                  structure_weight=0.6, color_weight=1.0, 
                  preserve_color=False, wplus=True, align_face_flag=True, 
                  predictor=None, truncation=0.7):
    
    # Align face if required
    if align_face_flag and predictor is not None:
        aligned_image = run_alignment(image_path, predictor)
        I = transform(aligned_image).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)
    
    # Process image
    with torch.no_grad():
        # Use the appropriate latent space based on wplus flag
        if wplus:
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                    z_plus_latent=False, return_z_plus_latent=False, resize=False)
        else:
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                    z_plus_latent=True, return_z_plus_latent=True, resize=False)
            
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        
        # Get latent
        latent = torch.tensor(exstyles[style_name]).repeat(2,1,1).to(device)
        
        # Preserve color if requested
        if preserve_color:
            latent[0:1,7:18] = instyle[0:1,7:18]
            latent[1,7:18] = instyle[0,7:18]
        elif color_weight < 1.0:
            latent[1,7:18] = instyle[0,7:18]
            
        # Get extrinsic style
        exstyle = generator.generator.style(
            latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])
        ).reshape(latent.shape)
        
        # Apply style transfer with weights
        interp_weights = [structure_weight]*7 + [color_weight]*11
        
        img_gen, _ = generator([instyle.repeat(2,1,1)], exstyle, z_plus_latent=not wplus, 
                             truncation=truncation, truncation_latent=0, use_res=True, 
                             interp_weights=interp_weights)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        
        # Apply style transfer with no color
        if not preserve_color and color_weight > 0:
            img_gen2, _ = generator([instyle], exstyle[0:1], z_plus_latent=not wplus, 
                                  truncation=truncation, truncation_latent=0, use_res=True, 
                                  interp_weights=[structure_weight]*7+[0]*11)
            img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)
            return img_rec, img_gen, img_gen2
        
    return img_rec, img_gen, None

# Function to create style weights grid
def create_style_weights_grid(image_path, generator, encoder, exstyles, style_name, 
                             wplus=True, align_face_flag=True, predictor=None, truncation=0.7):
    # Align face if required
    if align_face_flag and predictor is not None:
        aligned_image = run_alignment(image_path, predictor)
        I = transform(aligned_image).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)
    
    # Process image
    with torch.no_grad():
        if wplus:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                 z_plus_latent=False, return_z_plus_latent=False, resize=False)
        else:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                 z_plus_latent=True, return_z_plus_latent=True, resize=False)
        
        # Get latent
        latent = torch.tensor(exstyles[style_name]).repeat(1,1,1).to(device)
        exstyle = generator.generator.style(
            latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])
        ).reshape(latent.shape)
        
        results = []
        for i in range(6):  # Structure weights
            for j in range(6):  # Color weights
                w = [i/5.0]*7+[j/5.0]*11
                
                img_gen, _ = generator([instyle], exstyle, z_plus_latent=not wplus, 
                                    truncation=truncation, truncation_latent=0, use_res=True, 
                                    interp_weights=w)
                img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 128), -1, 1)
                results.append(img_gen)
        
        vis = torchvision.utils.make_grid(torch.cat(results, dim=0), 6, 1)
        return vis

# Function to generate a blended style
def create_style_blend(image_path, generator, encoder, exstyles, style_name1, style_name2, 
                      blend_weight=0.5, structure_weight=0.6, color_weight=1.0, 
                      wplus=True, align_face_flag=True, predictor=None, truncation=0.7):
    # Align face if required
    if align_face_flag and predictor is not None:
        aligned_image = run_alignment(image_path, predictor)
        I = transform(aligned_image).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)
    
    # Process image
    with torch.no_grad():
        if wplus:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                              z_plus_latent=False, return_z_plus_latent=False, resize=False)
        else:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                              z_plus_latent=True, return_z_plus_latent=True, resize=False)
        
        # Get and blend latents
        latent1 = torch.tensor(exstyles[style_name1]).unsqueeze(0).to(device)
        latent2 = torch.tensor(exstyles[style_name2]).unsqueeze(0).to(device)
        blended_latent = latent1 * blend_weight + latent2 * (1 - blend_weight)
        
        exstyle = generator.generator.style(
            blended_latent.reshape(blended_latent.shape[0]*blended_latent.shape[1], blended_latent.shape[2])
        ).reshape(blended_latent.shape)
        
        # Apply style transfer
        interp_weights = [structure_weight]*7 + [color_weight]*11
        img_gen, _ = generator([instyle], exstyle, z_plus_latent=not wplus, 
                             truncation=truncation, truncation_latent=0, use_res=True, 
                             interp_weights=interp_weights)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        
    return img_gen

# Function to generate a blend sequence
def create_blend_sequence(image_path, generator, encoder, exstyles, style_name1, style_name2, 
                         steps=6, structure_weight=0.6, color_weight=1.0, 
                         wplus=True, align_face_flag=True, predictor=None, truncation=0.7):
    # Align face if required
    if align_face_flag and predictor is not None:
        aligned_image = run_alignment(image_path, predictor)
        I = transform(aligned_image).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)
    
    # Process image
    with torch.no_grad():
        if wplus:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                              z_plus_latent=False, return_z_plus_latent=False, resize=False)
        else:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                              z_plus_latent=True, return_z_plus_latent=True, resize=False)
        
        # Get latents
        latent1 = torch.tensor(exstyles[style_name1]).repeat(steps,1,1).to(device)
        latent2 = torch.tensor(exstyles[style_name2]).repeat(steps,1,1).to(device)
        
        # Create blend weights
        fuse_weight = torch.arange(steps).reshape(steps,1,1).to(device) / (steps-1)
        fuse_latent = latent1 * fuse_weight + latent2 * (1-fuse_weight)
        
        exstyle = generator.generator.style(
            fuse_latent.reshape(fuse_latent.shape[0]*fuse_latent.shape[1], fuse_latent.shape[2])
        ).reshape(fuse_latent.shape)
        
        # Apply style transfer
        interp_weights = [structure_weight]*7 + [color_weight]*11
        img_gen, _ = generator([instyle.repeat(steps,1,1)], exstyle, z_plus_latent=not wplus, 
                            truncation=truncation, truncation_latent=0, use_res=True, 
                            interp_weights=interp_weights)
        img_gen = F.adaptive_avg_pool2d(torch.clamp(img_gen.detach(), -1, 1), 128)
        
        vis = torchvision.utils.make_grid(img_gen, steps, 1)
        return img_gen, vis

# Function to sample random styles
def sample_random_styles(generator, encoder, icptc, icpts, image_path, num_samples=6, 
                        structure_weight=0.6, color_weight=1.0, seed=None, 
                        wplus=True, align_face_flag=True, predictor=None, truncation=0.7):
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Align face if required
    if align_face_flag and predictor is not None:
        aligned_image = run_alignment(image_path, predictor)
        I = transform(aligned_image).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)
    
    # Process image
    with torch.no_grad():
        if wplus:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                              z_plus_latent=False, return_z_plus_latent=False, resize=False)
        else:
            _, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                              z_plus_latent=True, return_z_plus_latent=True, resize=False)
            
        instyle = instyle.repeat(num_samples, 1, 1)
        
        # Sample random style codes
        res_in = icpts.icp.netT(torch.randn(num_samples, 128).to(device)).reshape(-1, 7, 512)
        ada_in = icptc.icp.netT(torch.randn(num_samples, 128).to(device)).reshape(-1, 11, 512)
        
        # Concatenate codes
        latent = torch.cat((res_in, ada_in), dim=1)
        exstyle = generator.generator.style(
            latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])
        ).reshape(latent.shape)
        
        # Apply style transfer
        interp_weights = [structure_weight]*7 + [color_weight]*11
        img_gen, _ = generator([instyle], exstyle, z_plus_latent=not wplus, 
                            truncation=truncation, truncation_latent=0, use_res=True, 
                            interp_weights=interp_weights)
        img_gen = F.adaptive_avg_pool2d(torch.clamp(img_gen.detach(), -1, 1), 128)
        
        vis = torchvision.utils.make_grid(img_gen, num_samples, 1)
        return img_gen, vis

# Function to encode a custom style image
def encode_custom_style(generator, encoder, style_img_path, wplus=True):
    """
    Encode a custom style image into the DualStyleGAN latent space.
    
    Args:
        generator: The DualStyleGAN generator model
        encoder: The encoder model (pSp)
        style_img_path: Path to the style image
        wplus: Whether to use W+ latent space (True) or Z+ latent space (False)
        
    Returns:
        exstyle: The encoded style in the appropriate latent space, or None if failed
    """
    # Load and preprocess the style image
    if os.path.exists(style_img_path):
        try:
            S = load_image(style_img_path).to(device)
            
            # Encode the style image
            with torch.no_grad():
                if wplus:
                    _, style_codes = encoder(S, randomize_noise=False, return_latents=True, 
                                         z_plus_latent=False, return_z_plus_latent=False, resize=False)
                else:
                    _, style_codes = encoder(S, randomize_noise=False, return_latents=True, 
                                         z_plus_latent=True, return_z_plus_latent=True, resize=False)
                
                # Extract the style code
                exstyle = generator.generator.style(
                    style_codes.reshape(style_codes.shape[0]*style_codes.shape[1], style_codes.shape[2])
                ).reshape(style_codes.shape)
                
                return exstyle
        except Exception as e:
            print(f"Error encoding style image: {e}")
            return None
    
    return None

# Function to process image with custom style
def process_image_custom_style(image_path, style_img_paths, generator, encoder, 
                              structure_weight=0.6, color_weight=1.0, 
                              preserve_color=False, wplus=True, align_face_flag=True, 
                              predictor=None, blend_weights=None, truncation=0.7):
    
    # Align face if required
    if align_face_flag and predictor is not None:
        aligned_image = run_alignment(image_path, predictor)
        I = transform(aligned_image).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)
    
    # Process image
    with torch.no_grad():
        if wplus:
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                    z_plus_latent=False, return_z_plus_latent=False, resize=False)
        else:
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                    z_plus_latent=True, return_z_plus_latent=True, resize=False)
            
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        
        # Process each style image
        custom_styles = []
        for style_path in style_img_paths:
            style_code = encode_custom_style(generator, encoder, style_path, wplus)
            if style_code is not None:
                custom_styles.append(style_code)
        
        if not custom_styles:
            return img_rec, None, None
        
        # If multiple styles and blend weights are provided, blend them
        if len(custom_styles) > 1 and blend_weights:
            # Ensure weights sum to 1
            weights_sum = sum(blend_weights)
            normalized_weights = [w / weights_sum for w in blend_weights]
            
            # Initialize blended style with first weighted style
            blended_style = custom_styles[0] * normalized_weights[0]
            
            # Add remaining weighted styles
            for i in range(1, len(custom_styles)):
                if i < len(normalized_weights):
                    blended_style += custom_styles[i] * normalized_weights[i]
            
            exstyle = blended_style
        else:
            # Just use the first style
            exstyle = custom_styles[0]
        
        # Apply preserve color if requested
        if preserve_color:
            if len(exstyle.shape) == 3:  # For single exstyle
                exstyle_with_color = exstyle.clone()
                exstyle_with_color[0, 7:18] = instyle[0, 7:18]
                exstyle = exstyle_with_color
        
        # Apply style transfer with weights
        interp_weights = [structure_weight]*7 + [color_weight]*11
        
        # Repeat instyle if needed based on exstyle shape
        repeat_count = exstyle.shape[0]
        repeated_instyle = instyle.repeat(repeat_count, 1, 1)
        
        img_gen, _ = generator([repeated_instyle], exstyle, z_plus_latent=not wplus, 
                             truncation=truncation, truncation_latent=0, use_res=True, 
                             interp_weights=interp_weights)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        
        # Apply style transfer with no color
        if not preserve_color and color_weight > 0:
            img_gen2, _ = generator([instyle], exstyle[0:1], z_plus_latent=not wplus, 
                                  truncation=truncation, truncation_latent=0, use_res=True, 
                                  interp_weights=[structure_weight]*7+[0]*11)
            img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)
            return img_rec, img_gen, img_gen2
        
    return img_rec, img_gen, None

# Function to create latent space interpolation grid for custom style
def create_custom_style_interpolation_grid(image_path, style_img_path, generator, encoder, 
                                          options=None, align_face_flag=True, predictor=None):
    """
    Create a grid showing different latent space interpolations for a custom style.
    The grid explores different combinations of:
    - W+ vs Z+ latent space
    - Different structure/color weights
    - Different truncation values
    """
    # Default options
    if options is None:
        options = {
            'latent_spaces': ['w+', 'z+'],
            'structure_weights': [0.0, 0.3, 0.6, 1.0],
            'color_weights': [0.0, 0.3, 0.6, 1.0],
            'truncation_values': [0.5, 0.7, 0.9]
        }
    
    # Align face if required
    if align_face_flag and predictor is not None:
        aligned_image = run_alignment(image_path, predictor)
        I = transform(aligned_image).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)
    
    results = []
    labels = []
    
    # Process with different parameters
    with torch.no_grad():
        # For each latent space
        for latent_space in options['latent_spaces']:
            wplus = (latent_space == 'w+')
            
            # Encode content image
            if wplus:
                img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                       z_plus_latent=False, return_z_plus_latent=False, resize=False)
            else:
                img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True, 
                                       z_plus_latent=True, return_z_plus_latent=True, resize=False)
            
            # Encode style image
            custom_style = encode_custom_style(generator, encoder, style_img_path, wplus)
            
            if custom_style is None:
                continue
                
            # Try different structure and color weights
            for struct_weight in options['structure_weights']:
                for color_weight in options['color_weights']:
                    # Skip if both weights are 0
                    if struct_weight == 0.0 and color_weight == 0.0:
                        continue
                        
                    # For each truncation value
                    for trunc in options['truncation_values']:
                        interp_weights = [struct_weight] * 7 + [color_weight] * 11
                        
                        # Generate image with these parameters
                        img_gen, _ = generator([instyle], custom_style, z_plus_latent=not wplus, 
                                            truncation=trunc, truncation_latent=0, use_res=True, 
                                            interp_weights=interp_weights)
                        
                        img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 128), -1, 1)
                        results.append(img_gen)
                        
                        # Create label for this combination
                        label = f"{latent_space.upper()}, S:{struct_weight:.1f}, C:{color_weight:.1f}, T:{trunc:.1f}"
                        labels.append(label)
    
    # If no results, return None
    if not results:
        return None, None
        
    # Create grid of results
    grid = torchvision.utils.make_grid(torch.cat(results, dim=0), 
                                      nrow=len(options['color_weights']), 
                                      padding=2)
    return grid, labels

# Function to find optimal parameters for a custom style
def find_optimal_parameters(image_path, style_img_path, generator, encoder,
                           align_face_flag=True, predictor=None):
    """
    Try different parameter combinations to find the best settings for a custom style.
    Returns a sorted list of parameter sets from best to worst based on style transfer quality.
    """
    # This is a simplified approach without a true quality metric
    # In practice, you would need a more sophisticated evaluation method
    
    # Define parameter ranges to test
    test_params = {
        'latent_spaces': ['w+', 'z+'],
        'structure_weights': [0.4, 0.6, 0.8],
        'color_weights': [0.7, 0.9, 1.0],
        'truncation_values': [0.7]
    }
    
    # Creates a grid of parameter combinations
    grid, labels = create_custom_style_interpolation_grid(
        image_path, style_img_path, generator, encoder,
        test_params, align_face_flag, predictor
    )
    
    if grid is None:
        return []
    
    # In a real implementation, we would analyze the results and rank them
    # For now, we'll return a predefined ranking based on common patterns
    
    # Default recommended parameters (based on empirical observations)
    recommendations = [
        {'latent_space': 'w+', 'structure_weight': 0.6, 'color_weight': 0.9, 'truncation': 0.7},
        {'latent_space': 'w+', 'structure_weight': 0.8, 'color_weight': 1.0, 'truncation': 0.7},
        {'latent_space': 'w+', 'structure_weight': 0.4, 'color_weight': 0.7, 'truncation': 0.7},
        {'latent_space': 'z+', 'structure_weight': 0.6, 'color_weight': 0.9, 'truncation': 0.7},
        {'latent_space': 'z+', 'structure_weight': 0.8, 'color_weight': 1.0, 'truncation': 0.7}
    ]
    
    return recommendations, grid, labels

# Convert tensor to PIL image
def tensor_to_pil(tensor):
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).detach().numpy()
    return Image.fromarray((tensor * 255).astype(np.uint8))

# Convert tensor to bytes for download
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to check if all required checkpoints are available
def check_checkpoints():
    missing_files = []
    
    # Check encoder
    if not os.path.exists(os.path.join(MODEL_DIR, 'encoder.pt')):
        missing_files.append('encoder.pt')
    
    # Check face alignment model
    if not os.path.exists(os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')):
        print("Face alignment model not found. It will be downloaded automatically when needed.")
    
    # Check style-specific checkpoints
    for style_type in style_types:
        style_dir = os.path.join(MODEL_DIR, style_type)
        
        if not os.path.exists(os.path.join(style_dir, 'generator.pt')):
            missing_files.append(f'{style_type}/generator.pt')
        
        if not os.path.exists(os.path.join(style_dir, 'sampler.pt')):
            missing_files.append(f'{style_type}/sampler.pt')
        
        # Check for either refined_exstyle_code.npy or exstyle_code.npy
        if not (os.path.exists(os.path.join(style_dir, 'refined_exstyle_code.npy')) or
                os.path.exists(os.path.join(style_dir, 'exstyle_code.npy'))):
            missing_files.append(f'{style_type}/refined_exstyle_code.npy or exstyle_code.npy')
    
    # Check optional models
    optional_models = [
        'stylegan2-ffhq-config-f.pt',
        'model_ir_se50.pth'
    ]
    
    missing_optional = []
    for model in optional_models:
        if not os.path.exists(os.path.join(MODEL_DIR, model)):
            missing_optional.append(model)
    
    return missing_files, missing_optional
