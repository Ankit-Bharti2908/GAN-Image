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

# Create a UI header
def render_header():
    st.set_page_config(page_title="DualStyleGAN UI", page_icon="🎨", layout="wide")
    
    st.title("DualStyleGAN - Artistic Portrait Stylization")
    
    st.markdown("""
    This app allows you to transform portrait photos into artistic styles using DualStyleGAN.
    Upload your own photo and apply various artistic styles!
    """)
    
    # Check for required checkpoints
    missing_files, missing_optional = check_checkpoints()
    
    if missing_files:
        st.error("⚠️ Some required model files are missing:")
        for file in missing_files:
            st.write(f"- {file}")
        st.write("Please make sure all required models are downloaded before using this app.")
    
    if missing_optional:
        st.warning("Some optional model files are not found:")
        for file in missing_optional:
            st.write(f"- {file}")
        st.write("The app will still work but some advanced features might be limited.")

# Sidebar for model selection and information
def render_sidebar():
    st.sidebar.title("Style Settings")
    
    style_type = st.sidebar.selectbox(
        "Select style category",
        style_types,
        index=0
    )
    
    # Information sections
    st.sidebar.title("Information")
    
    with st.sidebar.expander("Parameter Information"):
        st.markdown("""
        ### Structure Weight
        Controls how strongly the structure (shape, geometry) of the style is applied.
        - **0.0**: No structural style applied
        - **1.0**: Full structural style applied
        
        ### Color Weight
        Controls how strongly the color palette of the style is applied.
        - **0.0**: No color style applied (original colors preserved)
        - **1.0**: Full color style applied
        
        ### Preserve Original Color
        When enabled, maintains the original image's colors while applying the style's structure.
        
        ### Use W+ Latent Space
        DualStyleGAN can use two different latent spaces:
        - **W+**: Usually produces better results with more flexibility
        - **Z+**: Original GAN latent space, sometimes necessary for specific styles
        
        ### Align Face
        Automatically detects and aligns faces in the portrait for better results.
        Disable if your image is already well-aligned or for non-face images.
        """)
    
    with st.sidebar.expander("About DualStyleGAN"):
        st.markdown("""
        ### DualStyleGAN
        
        DualStyleGAN is a deep learning model that can transform portrait photos into artistic styles.
        
        Features:
        - Separates style into extrinsic (artistic style) and intrinsic (identity) components
        - Allows fine control over structure and color transfer
        - Supports style blending and interpolation
        - Can generate random new styles using the trained style space
        
        [View on GitHub](https://github.com/williamyang1991/DualStyleGAN)
        
        Original paper: "DualStyleGAN: Image-to-Image Translation via Hierarchical Style Disentanglement"
        """)
                    
    with st.sidebar.expander("Usage Tips"):
        st.markdown("""
        ### Tips for Best Results
        
        1. **Image Selection**:
           - Use clear portrait photos
           - Frontal or slightly angled faces work best
           - Good lighting helps
        
        2. **Style Settings**:
           - Experiment with structure/color weights
           - Some styles work better with specific weights
           - Try the Style Weight Grid to find optimal settings
        
        3. **Face Alignment**:
           - Enable "Align Face" for best results with portraits
           - Disable for already aligned images or artistic effects
        
        4. **Style Blending**:
           - Blend similar styles for subtle variations
           - Blend contrasting styles for creative effects
           - Try different blend weights (0.3-0.7 often works well)
        """)
    
    with st.sidebar.expander("Custom Style Tips"):
        st.markdown("""
        ### Tips for Custom Styles
        
        **Best Style Images**:
        - Clear, distinctive artistic styles
        - Good lighting and focus
        - Strong color palettes
        - Recognizable artistic techniques
        
        **Intrinsic Model Selection**:
        - **W+ Latent**: Better for photorealistic styles
        - **Z+ Latent**: Better for abstract artistic styles
        
        **Troubleshooting**:
        - If results look distorted: Lower structure weight
        - If colors don't transfer well: Increase color weight
        - If identity is lost: Lower structure weight
        - If style doesn't apply strongly: Try the other latent model
        
        **Latent Space Exploration**:
        - Use the Latent Space Exploration tool to find optimal parameters
        - W+ generally works better for portrait photography styles
        - Z+ can work better for artistic or abstract styles
        - Different styles may need very different parameter settings
        """)
        
    with st.sidebar.expander("Understanding Latent Spaces"):
        st.markdown("""
        ### W+ vs Z+ Latent Space
        
        DualStyleGAN uses two different latent spaces for encoding:
        
        **W+ Latent Space**:
        - More disentangled representation
        - Better separation of identity and style
        - More control over specific features
        - Generally better for realistic styles and faces
        - Offers more fine-grained control over style aspects
        
        **Z+ Latent Space**:
        - The original GAN latent space
        - Sometimes captures artistic styles better
        - Can produce more dramatic transformations
        - May be better for abstract or heavily stylized art
        - Can sometimes produce more creative results
        
        **When to Use Each**:
        - Start with W+ for most portrait stylization
        - Try Z+ for abstract, cartoon, or artistic styles
        - Use the Latent Space Exploration to compare both
        - Some styles will have a clear preference for one space
        
        The optimal space depends on both the content and style image combination.
        """)
    
    return style_type

# Function to render the Single Image tab
def render_single_image_tab(style_type):
    st.header("Portrait Stylization")
    
    # Try to load models
    try:
        with st.spinner(f"Loading {style_type} models..."):
            generator, encoder, exstyles, icptc, icpts = load_models(style_type)
        
        style_names = get_style_names(style_type)
        if not style_names:
            st.error(f"No styles found for {style_type}")
            return
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Parameter selection
    col1, col2 = st.columns(2)
    
    with col1:
        style_id = st.selectbox(
            "Select style",
            range(len(style_names)),
            format_func=lambda x: f"Style {x}"
        )
        
        structure_weight = st.slider(
            "Structure Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6,
            help="Controls how strongly the structure (shape, geometry) of the style is applied"
        )
        
        color_weight = st.slider(
            "Color Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=1.0,
            help="Controls how strongly the color palette of the style is applied"
        )
        
        preserve_color = st.checkbox(
            "Preserve Original Color", 
            value=False,
            help="Maintain the original image's colors while applying the style's structure"
        )
    
    with col2:
        wplus = st.checkbox(
            "Use W+ latent space", 
            value=True,
            help="Uses the W+ latent space instead of Z+ (usually produces better results)"
        )
        
        align_face_flag = st.checkbox(
            "Align Face", 
            value=True,
            help="Automatically align face in the portrait for better results"
        )
        
        truncation = st.slider(
            "Truncation", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.7,
            help="Controls the diversity/quality tradeoff. Lower values = more regularized results"
        )
        
        # Try to show style image
        style_name = style_names[style_id]
        style_path = os.path.join(DATA_DIR, style_type, 'images/train', style_name)
        
        if os.path.exists(style_path):
            st.image(style_path, caption=f"Style: {style_name}", width=200)
        else:
            st.warning(f"Style image not found at {style_path}")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose a portrait image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        # Display the uploaded image
        st.image(temp_path, caption="Uploaded Image", width=300)
        
        # Process button
        if st.button("Apply Style"):
            try:
                # Get face predictor if needed
                predictor = get_dlib_predictor() if align_face_flag else None
                
                # Process image
                with st.spinner("Processing image..."):
                    img_rec, img_gen, img_gen2 = process_image(
                        temp_path, generator, encoder, exstyles, style_name,
                        structure_weight, color_weight, preserve_color,
                        wplus, align_face_flag, predictor, truncation
                    )
                
                # Display results
                st.subheader("Results")
                cols = st.columns(3 if img_gen2 is not None else 2)
                
                # Original reconstruction
                rec_pil = tensor_to_pil(img_rec)
                cols[0].image(rec_pil, caption="Original Reconstruction", width=300)
                cols[0].markdown(get_image_download_link(rec_pil, "original_rec.png", "Download Reconstruction"), unsafe_allow_html=True)
                
                # Full style transfer
                gen_pil = tensor_to_pil(img_gen[0:1])
                cols[1].image(gen_pil, caption="Style Transfer", width=300)
                cols[1].markdown(get_image_download_link(gen_pil, "styled.png", "Download Styled Image"), unsafe_allow_html=True)
                
                # Structure-only style transfer
                if img_gen2 is not None:
                    gen2_pil = tensor_to_pil(img_gen2)
                    cols[2].image(gen2_pil, caption="Structure Only", width=300)
                    cols[2].markdown(get_image_download_link(gen2_pil, "structure_only.png", "Download Structure Only"), unsafe_allow_html=True)
                
                # Save results
                output_path = os.path.join(OUTPUT_DIR, f"{style_type}_{style_id}.png")
                gen_pil.save(output_path)
                st.success(f"Saved result to {output_path}")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_path)

# Function to render the Style Weight Grid tab
def render_style_weight_grid_tab(style_type):
    st.header("Style Weight Grid")
    st.markdown("""
    This view shows different combinations of structure and color weights.
    - **Vertical axis**: Structure weight (0.0 to 1.0, bottom to top)
    - **Horizontal axis**: Color weight (0.0 to 1.0, left to right)
    """)
    
    # Try to load models
    try:
        with st.spinner(f"Loading {style_type} models..."):
            generator, encoder, exstyles, icptc, icpts = load_models(style_type)
        
        style_names = get_style_names(style_type)
        if not style_names:
            st.error(f"No styles found for {style_type}")
            return
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Parameter selection
    col1, col2 = st.columns(2)
    
    with col1:
        style_id_grid = st.selectbox(
            "Select style for grid",
            range(len(style_names)),
            format_func=lambda x: f"Style {x}",
            key="grid_style"
        )
        
        wplus_grid = st.checkbox(
            "Use W+ latent space", 
            value=True,
            key="grid_wplus",
            help="Uses the W+ latent space instead of Z+ (usually produces better results)"
        )
    
    with col2:
        align_face_grid = st.checkbox(
            "Align Face", 
            value=True,
            key="grid_align",
            help="Automatically align face in the portrait for better results"
        )
        
        truncation_grid = st.slider(
            "Truncation", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.7,
            key="grid_truncation",
            help="Controls the diversity/quality tradeoff"
        )
        
        # Try to show style image
        style_name_grid = style_names[style_id_grid]
        style_path_grid = os.path.join(DATA_DIR, style_type, 'images/train', style_name_grid)
        
        if os.path.exists(style_path_grid):
            st.image(style_path_grid, caption=f"Style: {style_name_grid}", width=200)
    
    # Image upload
    uploaded_file_grid = st.file_uploader("Choose a portrait image...", type=["jpg", "jpeg", "png"], key="grid_upload")
    
    if uploaded_file_grid is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file_grid.getvalue())
            temp_path_grid = tmp.name
        
        # Display the uploaded image
        st.image(temp_path_grid, caption="Uploaded Image", width=300)
        
        # Process button
        if st.button("Generate Style Grid"):
            try:
                # Get face predictor if needed
                predictor = get_dlib_predictor() if align_face_grid else None
                
                # Generate grid
                with st.spinner("Generating style weight grid..."):
                    grid = create_style_weights_grid(
                        temp_path_grid, generator, encoder, exstyles, style_name_grid,
                        wplus_grid, align_face_grid, predictor, truncation_grid
                    )
                
                # Display grid
                grid_pil = tensor_to_pil(grid)
                st.image(grid_pil, caption="Style Weight Grid", use_column_width=True)
                
                # Download link
                st.markdown(get_image_download_link(grid_pil, "style_grid.png", "Download Style Grid"), unsafe_allow_html=True)
                
                # Save grid
                output_path = os.path.join(OUTPUT_DIR, f"{style_type}_{style_id_grid}_grid.png")
                grid_pil.save(output_path)
                
            except Exception as e:
                st.error(f"Error generating grid: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_path_grid)

# Function to render the Style Blending tab
def render_style_blending_tab(style_type):
    st.header("Style Blending")
    st.markdown("""
    Blend two different styles together to create unique looks.
    You can also generate a sequence showing the transition between styles.
    """)
    
    # Try to load models
    try:
        with st.spinner(f"Loading {style_type} models..."):
            generator, encoder, exstyles, icptc, icpts = load_models(style_type)
        
        style_names = get_style_names(style_type)
        if not style_names:
            st.error(f"No styles found for {style_type}")
            return
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Parameter selection
    col1, col2 = st.columns(2)
    
    with col1:
        style_id1 = st.selectbox(
            "Style 1",
            range(len(style_names)),
            format_func=lambda x: f"Style {x}",
            key="blend_style1"
        )
        
        style_id2 = st.selectbox(
            "Style 2",
            range(len(style_names)),
            format_func=lambda x: f"Style {x}",
            key="blend_style2",
            index=min(1, len(style_names)-1)
        )
        
        blend_weight = st.slider(
            "Blend Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="Controls how much of Style 1 vs Style 2 to use (1.0 = all Style 1, 0.0 = all Style 2)"
        )
    
    with col2:
        structure_weight_blend = st.slider(
            "Structure Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6,
            key="blend_structure"
        )
        
        color_weight_blend = st.slider(
            "Color Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=1.0,
            key="blend_color"
        )
        
        wplus_blend = st.checkbox(
            "Use W+ latent space", 
            value=True,
            key="blend_wplus"
        )
        
        align_face_blend = st.checkbox(
            "Align Face", 
            value=True,
            key="blend_align"
        )
        
        truncation_blend = st.slider(
            "Truncation", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.7,
            key="blend_truncation"
        )
    
    # Try to show style images
    st.subheader("Selected Styles")
    col1, col2 = st.columns(2)
    
    style_name1 = style_names[style_id1]
    style_path1 = os.path.join(DATA_DIR, style_type, 'images/train', style_name1)
    
    style_name2 = style_names[style_id2]
    style_path2 = os.path.join(DATA_DIR, style_type, 'images/train', style_name2)
    
    if os.path.exists(style_path1):
        col1.image(style_path1, caption=f"Style 1: {style_name1}", width=200)
    
    if os.path.exists(style_path2):
        col2.image(style_path2, caption=f"Style 2: {style_name2}", width=200)
    
    # Image upload
    uploaded_file_blend = st.file_uploader("Choose a portrait image...", type=["jpg", "jpeg", "png"], key="blend_upload")
    
    if uploaded_file_blend is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file_blend.getvalue())
            temp_path_blend = tmp.name
        
        # Display the uploaded image
        st.image(temp_path_blend, caption="Uploaded Image", width=300)
        
        # Mode selection
        blend_mode = st.radio(
            "Blend Mode",
            ["Single Blend", "Transition Sequence"],
            index=0
        )
        
        if blend_mode == "Single Blend":
            if st.button("Generate Blended Style"):
                try:
                    # Get face predictor if needed
                    predictor = get_dlib_predictor() if align_face_blend else None
                    
                    # Generate blended style
                    with st.spinner("Generating blended style..."):
                        blended_img = create_style_blend(
                            temp_path_blend, generator, encoder, exstyles, 
                            style_name1, style_name2, blend_weight,
                            structure_weight_blend, color_weight_blend,
                            wplus_blend, align_face_blend, predictor, truncation_blend
                        )
                    
                    # Display result
                    blended_pil = tensor_to_pil(blended_img)
                    st.image(blended_pil, caption="Blended Style", width=400)
                    
                    # Download link
                    st.markdown(get_image_download_link(
                        blended_pil, 
                        f"blend_{style_id1}_{style_id2}_{blend_weight:.1f}.png", 
                        "Download Blended Image"
                    ), unsafe_allow_html=True)
                    
                    # Save result
                    output_path = os.path.join(
                        OUTPUT_DIR, 
                        f"blend_{style_type}_{style_id1}_{style_id2}_{blend_weight:.1f}.png"
                    )
                    blended_pil.save(output_path)
                    
                except Exception as e:
                    st.error(f"Error generating blended style: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(temp_path_blend)
        
        else:  # Transition Sequence
            steps = st.slider("Number of Steps", min_value=3, max_value=10, value=6)
            
            if st.button("Generate Transition Sequence"):
                try:
                    # Get face predictor if needed
                    predictor = get_dlib_predictor() if align_face_blend else None
                    
                    # Generate transition sequence
                    with st.spinner("Generating transition sequence..."):
                        sequence, sequence_grid = create_blend_sequence(
                            temp_path_blend, generator, encoder, exstyles, 
                            style_name1, style_name2, steps,
                            structure_weight_blend, color_weight_blend,
                            wplus_blend, align_face_blend, predictor, truncation_blend
                        )
                    
                    # Display transition grid
                    seq_grid_pil = tensor_to_pil(sequence_grid)
                    st.image(seq_grid_pil, caption="Style Transition Sequence", use_column_width=True)
                    
                    # Download link for grid
                    st.markdown(get_image_download_link(
                        seq_grid_pil, 
                        f"sequence_{style_id1}_{style_id2}.png", 
                        "Download Sequence Grid"
                    ), unsafe_allow_html=True)
                    
                    # Save sequence grid
                    output_path = os.path.join(
                        OUTPUT_DIR, 
                        f"sequence_{style_type}_{style_id1}_{style_id2}.png"
                    )
                    seq_grid_pil.save(output_path)
                    
                    # Show animation
                    st.subheader("Real-time Transformation Animation")
                    st.markdown("Watch how the style transitions from Style 1 to Style 2")
                    
                    # Create animation frames
                    frames = []
                    for i in range(steps):
                        frames.append(tensor_to_pil(sequence[i:i+1]))
                    
                    # Save frames for animation
                    temp_dir = tempfile.mkdtemp()
                    frame_paths = []
                    
                    for i, frame in enumerate(frames):
                        frame_path = os.path.join(temp_dir, f"frame_{i}.png")
                        frame.save(frame_path)
                        frame_paths.append(frame_path)
                    
                    # Display animation
                    animation_container = st.empty()
                    
                    # First display all frames
                    for frame_path in frame_paths:
                        animation_container.image(frame_path, width=300)
                        time.sleep(0.5)
                    
                    # Then loop animation a few times
                    for _ in range(3):
                        for frame_path in frame_paths:
                            animation_container.image(frame_path, width=300)
                            time.sleep(0.2)
                    
                    # Generate download link for animation frames
                    zip_path = os.path.join(OUTPUT_DIR, f"animation_{style_type}_{style_id1}_{style_id2}.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for frame_path in frame_paths:
                            zipf.write(frame_path, os.path.basename(frame_path))
                    
                    with open(zip_path, "rb") as f:
                        bytes_data = f.read()
                        b64 = base64.b64encode(bytes_data).decode()
                        href = f'<a href="data:file/zip;base64,{b64}" download="{os.path.basename(zip_path)}">Download Animation Frames</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating transition sequence: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(temp_path_blend)
                    
                    # Clean up frame directory
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass

# Function to render the Random Styles tab
def render_random_styles_tab(style_type):
    st.header("Random Styles")
    st.markdown("""
    Generate random styles using the sampler network.
    This can create unique styles that aren't in the original style set.
    """)
    
    # Try to load models
    try:
        with st.spinner(f"Loading {style_type} models..."):
            generator, encoder, exstyles, icptc, icpts = load_models(style_type)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Parameter selection
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.slider(
            "Number of Samples", 
            min_value=1, 
            max_value=12, 
            value=6
        )
        
        structure_weight_random = st.slider(
            "Structure Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6,
            key="random_structure"
        )
        
        color_weight_random = st.slider(
            "Color Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=1.0,
            key="random_color"
        )
    
    with col2:
        seed = st.number_input(
            "Random Seed", 
            min_value=0, 
            max_value=9999, 
            value=42,
            help="Set a specific seed for reproducible random styles"
        )
        
        wplus_random = st.checkbox(
            "Use W+ latent space", 
            value=True,
            key="random_wplus"
        )
        
        align_face_random = st.checkbox(
            "Align Face", 
            value=True,
            key="random_align"
        )
        
        truncation_random = st.slider(
            "Truncation", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.7,
            key="random_truncation"
        )
    
    # Image upload
    uploaded_file_random = st.file_uploader("Choose a portrait image...", type=["jpg", "jpeg", "png"], key="random_upload")
    
    if uploaded_file_random is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file_random.getvalue())
            temp_path_random = tmp.name
        
        # Display the uploaded image
        st.image(temp_path_random, caption="Uploaded Image", width=300)
        
        # Process button
        if st.button("Generate Random Styles"):
            try:
                # Get face predictor if needed
                predictor = get_dlib_predictor() if align_face_random else None
                
                # Generate random styles
                with st.spinner("Generating random styles..."):
                    samples, samples_grid = sample_random_styles(
                        generator, encoder, icptc, icpts, temp_path_random,
                        num_samples, structure_weight_random, color_weight_random,
                        seed, wplus_random, align_face_random, predictor, truncation_random
                    )
                
                # Display grid
                samples_grid_pil = tensor_to_pil(samples_grid)
                st.image(samples_grid_pil, caption="Random Styles", use_column_width=True)
                
                # Download link for grid
                st.markdown(get_image_download_link(
                    samples_grid_pil, 
                    f"random_styles_seed{seed}.png", 
                    "Download Random Styles Grid"
                ), unsafe_allow_html=True)
                
                # Save grid
                output_path = os.path.join(
                    OUTPUT_DIR, 
                    f"random_styles_{style_type}_seed{seed}.png"
                )
                samples_grid_pil.save(output_path)
                
                # Show individual images
                st.subheader("Individual Random Styles")
                
                # Display individual samples in columns
                cols = st.columns(min(3, num_samples))
                
                for i in range(num_samples):
                    col_idx = i % len(cols)
                    sample_pil = tensor_to_pil(samples[i:i+1])
                    
                    cols[col_idx].image(sample_pil, caption=f"Random Style {i+1}")
                    cols[col_idx].markdown(get_image_download_link(
                        sample_pil, 
                        f"random_style_{i+1}_seed{seed}.png", 
                        f"Download Style {i+1}"
                    ), unsafe_allow_html=True)
                    
                    # Save individual
                    output_path = os.path.join(
                        OUTPUT_DIR, 
                        f"random_style_{style_type}_{i+1}_seed{seed}.png"
                    )
                    sample_pil.save(output_path)
                
            except Exception as e:
                st.error(f"Error generating random styles: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_path_random)

# Function to render the Batch Processing tab
def render_batch_processing_tab(style_type):
    st.header("Batch Processing")
    st.markdown("""
    Process multiple images with the same style settings.
    Upload multiple images and apply the selected style to all of them at once.
    """)
    
    # Try to load models
    try:
        with st.spinner(f"Loading {style_type} models..."):
            generator, encoder, exstyles, icptc, icpts = load_models(style_type)
        
        style_names = get_style_names(style_type)
        if not style_names:
            st.error(f"No styles found for {style_type}")
            return
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Parameter selection
    col1, col2 = st.columns(2)
    
    with col1:
        style_id_batch = st.selectbox(
            "Select style",
            range(len(style_names)),
            format_func=lambda x: f"Style {x}",
            key="batch_style"
        )
        
        structure_weight_batch = st.slider(
            "Structure Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6,
            key="batch_structure"
        )
        
        color_weight_batch = st.slider(
            "Color Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=1.0,
            key="batch_color"
        )
    
    with col2:
        preserve_color_batch = st.checkbox(
            "Preserve Original Color", 
            value=False,
            key="batch_preserve"
        )
        
        wplus_batch = st.checkbox(
            "Use W+ latent space", 
            value=True,
            key="batch_wplus"
        )
        
        align_face_batch = st.checkbox(
            "Align Face", 
            value=True,
            key="batch_align"
        )
        
        truncation_batch = st.slider(
            "Truncation", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.7,
            key="batch_truncation"
        )
    
    # Try to show style image
    style_name_batch = style_names[style_id_batch]
    style_path_batch = os.path.join(DATA_DIR, style_type, 'images/train', style_name_batch)
    
    if os.path.exists(style_path_batch):
        st.image(style_path_batch, caption=f"Style: {style_name_batch}", width=200)
    
    # Multiple file upload
    uploaded_files_batch = st.file_uploader(
        "Choose portrait images...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files_batch:
        st.write(f"Uploaded {len(uploaded_files_batch)} images")
        
        # Display thumbnails of uploaded images
        cols = st.columns(min(4, len(uploaded_files_batch)))
        
        for i, uploaded_file in enumerate(uploaded_files_batch[:4]):  # Show first 4 images
            col_idx = i % len(cols)
            cols[col_idx].image(uploaded_file, caption=f"Image {i+1}", width=150)
        
        if len(uploaded_files_batch) > 4:
            st.write(f"... and {len(uploaded_files_batch) - 4} more")
        
        # Process button
        if st.button("Process Batch"):
            try:
                # Create temp directory for batch processing
                batch_temp_dir = tempfile.mkdtemp()
                batch_output_dir = os.path.join(OUTPUT_DIR, f"batch_{style_type}_{style_id_batch}_{int(time.time())}")
                os.makedirs(batch_output_dir, exist_ok=True)
                
                # Get face predictor if needed
                predictor = get_dlib_predictor() if align_face_batch else None
                
                # Process each image
                with st.spinner(f"Processing {len(uploaded_files_batch)} images..."):
                    progress_bar = st.progress(0)
                    
                    for i, uploaded_file in enumerate(uploaded_files_batch):
                        # Save the uploaded file temporarily
                        temp_path_batch = os.path.join(batch_temp_dir, f"image_{i}.jpg")
                        with open(temp_path_batch, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Process image
                        img_rec, img_gen, _ = process_image(
                            temp_path_batch, generator, encoder, exstyles, style_name_batch,
                            structure_weight_batch, color_weight_batch, preserve_color_batch,
                            wplus_batch, align_face_batch, predictor, truncation_batch
                        )
                        
                        # Save styled image
                        styled_pil = tensor_to_pil(img_gen[0:1])
                        output_path = os.path.join(batch_output_dir, f"styled_image_{i}.png")
                        styled_pil.save(output_path)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files_batch))
                
                # Create zip file of results
                zip_path = os.path.join(OUTPUT_DIR, f"batch_results_{style_type}_{style_id_batch}.zip")
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file in os.listdir(batch_output_dir):
                        file_path = os.path.join(batch_output_dir, file)
                        zipf.write(file_path, os.path.basename(file_path))
                
                # Create download link for zip
                with open(zip_path, "rb") as f:
                    bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()
                    href = f'<a href="data:file/zip;base64,{b64}" download="{os.path.basename(zip_path)}">Download All Processed Images</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # Show results
                st.subheader("Batch Processing Results")
                
                # Display grid of results (first 6 images)
                result_files = sorted([os.path.join(batch_output_dir, f) for f in os.listdir(batch_output_dir)])[:6]
                result_cols = st.columns(min(3, len(result_files)))
                
                for i, result_file in enumerate(result_files):
                    col_idx = i % len(result_cols)
                    result_cols[col_idx].image(result_file, caption=f"Result {i+1}", width=200)
                
                if len(result_files) < len(uploaded_files_batch):
                    st.write(f"... and {len(uploaded_files_batch) - len(result_files)} more in the zip file")
                
            except Exception as e:
                st.error(f"Error in batch processing: {str(e)}")
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(batch_temp_dir)
                except:
                    pass

# Function to render the Custom Style tab
def render_custom_style_tab(style_type):
    st.header("Custom Style Images")
    st.markdown("""
    Upload your own style images and apply them to your portraits.
    You can upload one style image or multiple for blending.
    """)
    
    # Try to load models
    try:
        with st.spinner(f"Loading {style_type} models..."):
            generator, encoder, exstyles, icptc, icpts = load_models(style_type)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Create tabs for different custom style features
    custom_tab1, custom_tab2 = st.tabs(["Basic Custom Style", "Latent Space Exploration"])
    
    with custom_tab1:
        # Parameter selection
        col1, col2 = st.columns(2)
        
        with col1:
            structure_weight_custom = st.slider(
                "Structure Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6,
                key="custom_structure"
            )
            
            color_weight_custom = st.slider(
                "Color Weight", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.9,
                key="custom_color"
            )
            
            preserve_color_custom = st.checkbox(
                "Preserve Original Color", 
                value=False,
                key="custom_preserve"
            )
        
        with col2:
            truncation_custom = st.slider(
                "Truncation", 
                min_value=0.5, 
                max_value=1.0, 
                value=0.7,
                help="Controls the diversity/quality tradeoff. Lower values = more regularized results"
            )
            
            align_face_custom = st.checkbox(
                "Align Face", 
                value=True,
                key="custom_align"
            )
            
            intrinsic_model = st.radio(
                "Intrinsic Model",
                ["W+ Latent", "Z+ Latent"],
                index=0,
                help="W+ is generally better for faces, Z+ can be better for certain artistic styles"
            )
        
        # Set wplus based on intrinsic model selection
        wplus_custom = (intrinsic_model == "W+ Latent")
        
        # Style image upload
        st.subheader("Upload Style Image(s)")
        
        style_mode = st.radio(
            "Style Mode",
            ["Single Style", "Multiple Styles (Blend)"],
            index=0
        )
        
        if style_mode == "Single Style":
            uploaded_style = st.file_uploader(
                "Choose a style image...", 
                type=["jpg", "jpeg", "png"], 
                key="custom_style_upload"
            )
            
            if uploaded_style is not None:
                # Save the style image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_style.getvalue())
                    style_path = tmp.name
                
                # Display the style image
                st.image(style_path, caption="Uploaded Style Image", width=200)
                style_paths = [style_path]
                style_weights = [1.0]
            else:
                style_paths = []
                style_weights = []
                
        else:  # Multiple Styles (Blend)
            uploaded_styles = st.file_uploader(
                "Choose style images...", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True,
                key="custom_styles_upload"
            )
            
            style_paths = []
            style_weights = []
            
            if uploaded_styles:
                st.write(f"Uploaded {len(uploaded_styles)} style images")
                
                # Display style images with weight sliders
                for i, uploaded_style in enumerate(uploaded_styles):
                    col1, col2 = st.columns([1, 2])
                    
                    # Save the style image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_style.getvalue())
                        style_path = tmp.name
                    
                    style_paths.append(style_path)
                    
                    # Display the style image
                    col1.image(style_path, caption=f"Style {i+1}", width=150)
                    
                    # Weight slider
                    weight = col2.slider(
                        f"Weight for Style {i+1}", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=1.0 / len(uploaded_styles),
                        key=f"weight_{i}"
                    )
                    style_weights.append(weight)
                
                # Normalize weights visually
                total_weight = sum(style_weights)
                if total_weight > 0:
                    st.write("Normalized Weights:")
                    normalized_weights = [w / total_weight for w in style_weights]
                    
                    # Create a visual representation of weights
                    weight_cols = st.columns(len(style_weights))
                    for i, (w, nw) in enumerate(zip(style_weights, normalized_weights)):
                        weight_cols[i].metric(f"Style {i+1}", f"{nw:.2f}")
        
        # Content image upload
        st.subheader("Upload Content Image")
        uploaded_content = st.file_uploader(
            "Choose a portrait image...", 
            type=["jpg", "jpeg", "png"], 
            key="custom_content_upload"
        )
        
        if uploaded_content is not None and style_paths:
            # Save the content image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_content.getvalue())
                content_path = tmp.name
            
            # Display the content image
            st.image(content_path, caption="Content Image", width=300)
            
            # Process button
            if st.button("Apply Custom Style"):
                try:
                    # Get face predictor if needed
                    predictor = get_dlib_predictor() if align_face_custom else None
                    
                    # Process image with custom style(s)
                    with st.spinner("Processing image with custom style..."):
                        img_rec, img_gen, img_gen2 = process_image_custom_style(
                            content_path, style_paths, generator, encoder,
                            structure_weight_custom, color_weight_custom, preserve_color_custom,
                            wplus_custom, align_face_custom, predictor, style_weights, truncation_custom
                        )
                    
                    if img_gen is None:
                        st.error("Failed to process custom style. Please try a different style image.")
                    else:
                        # Display results
                        st.subheader("Results")
                        cols = st.columns(3 if img_gen2 is not None else 2)
                        
                        # Original reconstruction
                        rec_pil = tensor_to_pil(img_rec)
                        cols[0].image(rec_pil, caption="Original Reconstruction", width=300)
                        cols[0].markdown(get_image_download_link(rec_pil, "original_rec.png", "Download Reconstruction"), unsafe_allow_html=True)
                        
                        # Custom style transfer
                        gen_pil = tensor_to_pil(img_gen[0:1])
                        cols[1].image(gen_pil, caption="Custom Style Transfer", width=300)
                        cols[1].markdown(get_image_download_link(gen_pil, "custom_styled.png", "Download Styled Image"), unsafe_allow_html=True)
                        
                        # Structure-only style transfer
                        if img_gen2 is not None:
                            gen2_pil = tensor_to_pil(img_gen2)
                            cols[2].image(gen2_pil, caption="Structure Only", width=300)
                            cols[2].markdown(get_image_download_link(gen2_pil, "custom_structure_only.png", "Download Structure Only"), unsafe_allow_html=True)
                        
                        # Save results
                        output_name = f"custom_style_{int(time.time())}.png"
                        output_path = os.path.join(OUTPUT_DIR, output_name)
                        gen_pil.save(output_path)
                        st.success(f"Saved result to {output_path}")
                    
                except Exception as e:
                    st.error(f"Error processing image with custom style: {str(e)}")
                finally:
                    # Clean up temporary files
                    for path in style_paths + [content_path]:
                        try:
                            os.unlink(path)
                        except:
                            pass
    
    with custom_tab2:
        st.subheader("Latent Space Exploration")
        st.markdown("""
        This tool helps you find the optimal latent space settings for your custom style.
        Upload a style image and a content image, and the system will generate a grid of 
        different parameter combinations to help you identify the best settings.
        """)
        
        # Style image upload for exploration
        uploaded_style_explore = st.file_uploader(
            "Choose a style image...", 
            type=["jpg", "jpeg", "png"], 
            key="explore_style_upload"
        )
        
        style_path_explore = None
        if uploaded_style_explore is not None:
            # Save the style image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_style_explore.getvalue())
                style_path_explore = tmp.name
            
            # Display the style image
            st.image(style_path_explore, caption="Style Image for Exploration", width=200)
        
        # Content image upload for exploration
        uploaded_content_explore = st.file_uploader(
            "Choose a portrait image...", 
            type=["jpg", "jpeg", "png"], 
            key="explore_content_upload"
        )
        
        content_path_explore = None
        if uploaded_content_explore is not None:
            # Save the content image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_content_explore.getvalue())
                content_path_explore = tmp.name
            
            # Display the content image
            st.image(content_path_explore, caption="Content Image for Exploration", width=200)
        
        # Exploration options
        with st.expander("Advanced Exploration Settings"):
            exploration_mode = st.radio(
                "Exploration Mode",
                ["Quick (Recommended Parameters)", "Comprehensive (All Combinations)"],
                index=0
            )
            
            if exploration_mode == "Comprehensive (All Combinations)":
                st.warning("Comprehensive exploration will test many combinations and may take several minutes.")
                
                # Allow user to customize exploration parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    latent_spaces = st.multiselect(
                        "Latent Spaces",
                        ["w+", "z+"],
                        default=["w+", "z+"]
                    )
                    
                    structure_weights = [float(x) for x in st.text_input(
                        "Structure Weights (comma-separated)",
                        value="0.0,0.3,0.6,1.0"
                    ).split(",")]
                
                with col2:
                    color_weights = [float(x) for x in st.text_input(
                        "Color Weights (comma-separated)",
                        value="0.0,0.3,0.6,1.0"
                    ).split(",")]
                    
                    truncation_values = [float(x) for x in st.text_input(
                        "Truncation Values (comma-separated)",
                        value="0.5,0.7,0.9"
                    ).split(",")]
                
                exploration_params = {
                    'latent_spaces': latent_spaces,
                    'structure_weights': structure_weights,
                    'color_weights': color_weights,
                    'truncation_values': truncation_values
                }
            else:
                # Default recommended parameter sets for quick exploration
                exploration_params = {
                    'latent_spaces': ['w+', 'z+'],
                    'structure_weights': [0.4, 0.6, 0.8],
                    'color_weights': [0.7, 0.9, 1.0],
                    'truncation_values': [0.7]
                }
        
        # Align face option
        align_face_explore = st.checkbox(
            "Align Face", 
            value=True,
            key="explore_align"
        )
        
        # Explore button
        if style_path_explore and content_path_explore:
            if st.button("Explore Latent Space"):
                try:
                    # Get face predictor if needed
                    predictor = get_dlib_predictor() if align_face_explore else None
                    
                    # Generate the interpolation grid
                    with st.spinner("Exploring latent space combinations. This may take a moment..."):
                        if exploration_mode == "Quick (Recommended Parameters)":
                            # Use the find_optimal_parameters function for quick exploration
                            recommendations, grid, labels = find_optimal_parameters(
                                content_path_explore, style_path_explore, 
                                generator, encoder, align_face_explore, predictor
                            )
                        else:
                            # Use the create_custom_style_interpolation_grid function for comprehensive exploration
                            grid, labels = create_custom_style_interpolation_grid(
                                content_path_explore, style_path_explore,
                                generator, encoder, exploration_params,
                                align_face_explore, predictor
                            )
                            recommendations = []
                    
                    if grid is None:
                        st.error("Failed to generate exploration grid. Please try different images.")
                    else:
                        # Display the interpolation grid
                        st.subheader("Latent Space Exploration Results")
                        
                        # Convert grid to PIL and display
                        grid_pil = tensor_to_pil(grid)
                        st.image(grid_pil, caption="Parameter Exploration Grid", use_column_width=True)
                        
                        # Download link for grid
                        st.markdown(get_image_download_link(
                            grid_pil, 
                            "latent_space_exploration.png", 
                            "Download Exploration Grid"
                        ), unsafe_allow_html=True)
                        
                        # Show the labels/parameters for each image in the grid
                        with st.expander("Parameter Details for Each Image in Grid"):
                            for i, label in enumerate(labels):
                                st.text(f"Image {i+1}: {label}")
                        
                        # Display recommended parameters (if available)
                        if recommendations:
                            st.subheader("Recommended Parameter Settings")
                            st.write("Based on our analysis, these parameter combinations should work well with your custom style:")
                            
                            for i, rec in enumerate(recommendations):
                                st.write(f"**Option {i+1}:** Latent Space: {rec['latent_space'].upper()}, "
                                       f"Structure: {rec['structure_weight']}, "
                                       f"Color: {rec['color_weight']}, "
                                       f"Truncation: {rec['truncation']}")
                            
                            # Option to apply a recommended setting
                            st.write("---")
                            rec_to_apply = st.selectbox(
                                "Apply a recommended setting",
                                options=range(len(recommendations)),
                                format_func=lambda x: f"Option {x+1}"
                            )
                            
                            selected_rec = recommendations[rec_to_apply]
                            
                            if st.button("Apply Selected Parameters"):
                                # Apply the selected parameters to the custom style
                                wplus_selected = (selected_rec['latent_space'] == 'w+')
                                struct_weight_selected = selected_rec['structure_weight']
                                color_weight_selected = selected_rec['color_weight']
                                trunc_selected = selected_rec['truncation']
                                
                                with st.spinner("Applying selected parameters..."):
                                    try:
                                        # Process image with these parameters
                                        img_rec, img_gen, _ = process_image_custom_style(
                                            content_path_explore, [style_path_explore], 
                                            generator, encoder, struct_weight_selected, 
                                            color_weight_selected, False, wplus_selected, 
                                            align_face_explore, predictor, [1.0], trunc_selected
                                        )
                                        
                                        # Display result
                                        st.subheader("Result with Selected Parameters")
                                        cols = st.columns(2)
                                        
                                        # Original reconstruction
                                        rec_pil = tensor_to_pil(img_rec)
                                        cols[0].image(rec_pil, caption="Original", width=300)
                                        
                                        # Styled result
                                        gen_pil = tensor_to_pil(img_gen[0:1])
                                        cols[1].image(gen_pil, caption=f"Applied Style with {selected_rec['latent_space'].upper()} parameters", width=300)
                                        cols[1].markdown(get_image_download_link(
                                            gen_pil, 
                                            f"optimized_style_{selected_rec['latent_space']}.png", 
                                            "Download Optimized Result"
                                        ), unsafe_allow_html=True)
                                        
                                    except Exception as e:
                                        st.error(f"Error applying parameters: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error exploring latent space: {str(e)}")
                finally:
                    # Clean up temporary files
                    try:
                        os.unlink(style_path_explore)
                        os.unlink(content_path_explore)
                    except:
                        pass
        
        # Information about latent space exploration
        with st.expander("Understanding Latent Space Exploration"):
            st.markdown("""
            ### What is Latent Space Exploration?
            
            The latent space in DualStyleGAN contains the encoded representations of both content and style. 
            Finding the right combination of parameters in this space can dramatically improve results for custom styles.
            
            ### Key Parameters to Explore:
            
            1. **Latent Space Type**:
               - **W+ Latent**: The extended latent space with more control over specific features
               - **Z+ Latent**: The original latent space that can work better for more abstract styles
            
            2. **Structure Weight**:
               - Controls how strongly geometric aspects of the style are applied
               - Higher values (0.8-1.0) transfer more of the style's structural elements
               - Lower values (0.2-0.4) retain more of the original face structure
            
            3. **Color Weight**:
               - Controls how strongly color palettes and textures are applied
               - Higher values (0.8-1.0) transfer the style's colors/textures more intensely
               - Lower values (0.2-0.4) preserve more of the original image's colors
            
            4. **Truncation**:
               - Controls the diversity/quality trade-off in the generated images
               - Lower values (0.5-0.7) create more regularized, averaged results
               - Higher values (0.8-1.0) allow more diverse outputs that may be less stable
            
            ### How to Read the Results Grid:
            
            The exploration grid shows combinations of these parameters. When interpreting results:
            
            1. Look for combinations where facial identity is preserved but style is clearly visible
            2. Notice how different latent spaces affect the results (W+ vs Z+)
            3. Pay attention to how structure vs color weights change the outcome
            4. Use the recommended settings as starting points, then fine-tune
            
            ### Tips for Finding the Best Parameters:
            
            - **For Portraits/Faces**: Start with W+ latent space and higher structure weights (0.6+)
            - **For Abstract Styles**: Try Z+ latent space with balanced structure/color weights
            - **For Preserving Identity**: Lower structure weights (0.3-0.5) and higher color weights
            - **For Strong Style Transfer**: Higher structure weights (0.7-1.0) with balanced color
            
            The exploration helps you understand which combinations work best for your specific 
            style image and content combination.
            """)
    
    # Information about custom styles
    with st.expander("About Custom Style Transfer"):
        st.markdown("""
        ### Using Custom Style Images
        
        DualStyleGAN can use your own images as style references. Here's how it works:
        
        1. **Extrinsic Style**:
           - Your uploaded style image(s) are encoded to extract their artistic style
           - These become the "extrinsic style codes" that define the artistic look
        
        2. **Intrinsic Style**:
           - Your content portrait's identity and basic structure is preserved
           - The model balances between preserving identity and applying the new style
        
        3. **Latent Space Models**:
           - **W+ Latent**: Usually works better for realistic faces and detailed styles
           - **Z+ Latent**: Can work better for more abstract or artistic styles
        
        4. **Style Blending**:
           - Upload multiple style images to create a unique blended style
           - Adjust the weight sliders to control how much each style contributes
           - The weights are normalized automatically (they will sum to 1.0)
        
        5. **Tips for Custom Styles**:
           - Clear, high-quality style images work best
           - Styles with distinctive artistic elements tend to transfer better
           - Experiment with different structure/color weights for best results
           - If the result is too distorted, try lowering the structure weight
        """)
        
        st.markdown("""
        ### Technical Details
        
        The custom style process works as follows:
        
        1. Your style image is encoded using the same encoder used for content images
        2. This creates a latent code in either W+ or Z+ space (based on your selection)
        3. The code is transformed into the extended style space of DualStyleGAN
        4. For multiple styles, the codes are blended using weighted averaging
        5. The style transfer is applied with your selected structure and color weights
        
        For best results with custom styles, you may need to experiment with different
        parameters, especially the structure weight and intrinsic model selection.
        """)

# Main function to run the Streamlit app
def run_app():
    render_header()
    style_type = render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Single Image", 
        "Style Weight Grid", 
        "Style Blending", 
        "Random Styles", 
        "Batch Processing",
        "Custom Style"
    ])
    
    with tab1:
        render_single_image_tab(style_type)
    
    with tab2:
        render_style_weight_grid_tab(style_type)
    
    with tab3:
        render_style_blending_tab(style_type)
    
    with tab4:
        render_random_styles_tab(style_type)
    
    with tab5:
        render_batch_processing_tab(style_type)
    
    with tab6:
        render_custom_style_tab(style_type)

if __name__ == "__main__":
    run_app()
