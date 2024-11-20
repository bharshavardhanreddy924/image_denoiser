import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd

from rich.console import Console
from rich.table import Table

# Direct import since files are in same directory
from pipeline import ISPPipeline, DenoiseSharpenPipeline

def convert_image_for_display(image):
    """Convert image for Streamlit display"""
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    
    return image

def main():
    st.set_page_config(page_title="Advanced Image Signal Processing", layout="wide")
    
    st.title("🖼️ Advanced Image Signal Processing Pipeline")
    
    # Sidebar for configuration
    st.sidebar.header("Image Processing Options")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload RAW Image", 
        type=['raw'], 
        help="Upload a 12-bit RAW image in GRBG Bayer pattern"
    )
    
    # Processing methods selection
    processing_options = st.sidebar.multiselect(
        "Select Processing Methods",
        [
            "Demosaic", 
            "White Balance", 
            "Gamma Correction", 
            "Gaussian Denoising", 
            "Median Denoising", 
            "Bilateral Denoising", 
            "U-Net Denoising",
            "Unsharp Mask", 
            "Laplacian Sharpening"
        ],
        default=["Demosaic", "White Balance", "Gamma Correction"]
    )
    
    # ROI selection
    st.sidebar.subheader("Region of Interest")
    roi_x = st.sidebar.slider("ROI X Position", 0, 1920, 200)
    roi_y = st.sidebar.slider("ROI Y Position", 0, 1280, 200)
    roi_width = st.sidebar.slider("ROI Width", 100, 800, 400)
    roi_height = st.sidebar.slider("ROI Height", 100, 800, 400)
    
    if uploaded_file is not None:
        # Initialize pipelines
        isp_pipeline = ISPPipeline()
        denoiser_pipeline = DenoiseSharpenPipeline()
        
        # Save uploaded file temporarily
        with open("temp_upload.raw", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read RAW image
        raw_image = isp_pipeline.read_raw("temp_upload.raw")
        
        # Image processing steps
        processed_images = {}
        
        if "Demosaic" in processing_options:
            demosaiced = isp_pipeline.demosaic(raw_image)
            processed_images["Demosaic"] = convert_image_for_display(demosaiced)
        
        if "White Balance" in processing_options:
            wb_image = isp_pipeline.white_balance(demosaiced)
            processed_images["White Balance"] = convert_image_for_display(wb_image)
        
        if "Gamma Correction" in processing_options:
            gamma_image = isp_pipeline.apply_gamma(wb_image)
            processed_images["Gamma Correction"] = convert_image_for_display(gamma_image)
        
        # Denoising methods
        denoising_methods = [
            "Gaussian Denoising", 
            "Median Denoising", 
            "Bilateral Denoising", 
            "U-Net Denoising"
        ]
        
        if any(method in processing_options for method in denoising_methods):
            denoised_results = denoiser_pipeline.apply_denoise_methods(gamma_image)
            for method, img in denoised_results.items():
                if method != 'original' and method.replace(' ', '_').lower() in [x.replace(' ', '_').lower() for x in processing_options]:
                    processed_images[f"Denoising: {method.title()}"] = img
        
        # Sharpening methods
        sharpening_methods = ["Unsharp Mask", "Laplacian Sharpening"]
        if any(method in processing_options for method in sharpening_methods):
            sharpened_results = denoiser_pipeline.apply_sharpen_methods(gamma_image)
            for method, img in sharpened_results.items():
                processed_images[f"Sharpening: {method.replace('_', ' ').title()}"] = img
        
        # Metrics computation
        roi = (roi_x, roi_y, roi_width, roi_height)
        metrics_results = {}
        
        for name, img in processed_images.items():
            snr, edge_strength = denoiser_pipeline.compute_metrics(img, roi)
            metrics_results[name] = {
                'SNR': snr,
                'Edge Strength': edge_strength
            }
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Processed Images")
            for name, img in processed_images.items():
                st.subheader(name)
                st.image(img, channels="RGB")
        
        with col2:
            st.header("Image Quality Metrics")
            metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index')
            st.dataframe(metrics_df)
        
        # ROI visualization
        st.header("Regions of Interest")
        for name, img in processed_images.items():
            img_with_roi = img.copy()
            cv2.rectangle(
                img_with_roi, 
                (roi_x, roi_y), 
                (roi_x + roi_width, roi_y + roi_height), 
                (0, 255, 0), 
                2
            )
            st.image(img_with_roi, caption=f"ROI for {name}", channels="RGB")

if __name__ == "__main__":
    main()