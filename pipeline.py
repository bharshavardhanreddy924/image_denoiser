import cv2
import numpy as np
import tensorflow as tf
import os
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ISPPipeline:
    """Basic Image Signal Processing Pipeline"""
    
    def __init__(self):
        self.console = Console()
        
    def read_raw(self, filepath, width=1920, height=1280):
        """
        Read 12-bit RAW image in GRBG Bayer pattern
        Args:
            filepath: Path to RAW file
            width: Image width (default 1920)
            height: Image height (default 1280)
        Returns:
            numpy array of shape (height, width) with 16-bit values
        """
        try:
            with open(filepath, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint16)
                
            if len(raw_data) != width * height:
                raise ValueError(f"Expected {width*height} pixels, got {len(raw_data)}")
                
            # Reshape and mask to 12 bits
            raw_image = raw_data.reshape((height, width))
            raw_image = raw_image & 0x0FFF
            raw_image = raw_image.astype(np.uint16)
            
            # Scale to full 16-bit range
            raw_image = (raw_image << 4).astype(np.uint16)
            return raw_image
            
        except Exception as e:
            self.console.print(f"[red]Error loading RAW image: {e}[/red]")
            return None
            
    def demosaic(self, bayer_image):
        """
        Edge-aware demosaicing for GRBG Bayer pattern
        """
        try:
            return cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2RGB)
        except Exception as e:
            self.console.print(f"[red]Demosaicing error: {e}[/red]")
            return None
            
    def white_balance(self, image):
        """
        Apply gray world white balance
        """
        try:
            result = np.zeros_like(image)
            for i in range(3):
                avg = np.mean(image[:, :, i])
                result[:, :, i] = np.clip(image[:, :, i] * (128 / avg), 0, 65535)
            return result
        except Exception as e:
            self.console.print(f"[red]White balance error: {e}[/red]")
            return None
            
    def apply_gamma(self, image, gamma=2.2):
        """
        Apply gamma correction and convert to 8-bit
        """
        try:
            normalized = image.astype(np.float32) / 65535.0
            gamma_corrected = np.power(normalized, 1/gamma)
            return (gamma_corrected * 255).astype(np.uint8)
        except Exception as e:
            self.console.print(f"[red]Gamma correction error: {e}[/red]")
            return None

class DenoiseSharpenPipeline:
    """Advanced denoising and sharpening pipeline"""
    
    def __init__(self):
        self.console = Console()
        self.isp = ISPPipeline()
        self.unet = self._build_unet()
        
    def _build_unet(self):
        """Build U-Net model for denoising"""
        try:
            inputs = tf.keras.layers.Input(shape=(None, None, 3))
            
            # Encoder
            conv1 = self._conv_block(inputs, 64)
            pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
            
            conv2 = self._conv_block(pool1, 128)
            pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
            
            # Bridge
            conv3 = self._conv_block(pool2, 256)
            
            # Decoder
            up4 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
            up4 = tf.keras.layers.Concatenate()([up4, conv2])
            conv4 = self._conv_block(up4, 128)
            
            up5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
            up5 = tf.keras.layers.Concatenate()([up5, conv1])
            conv5 = self._conv_block(up5, 64)
            
            outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(conv5)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            try:
                model.load_weights('unet_denoiser_weights.h5')
                self.console.print("[green]Loaded pretrained U-Net weights[/green]")
            except:
                self.console.print("[yellow]No pretrained weights found. Using initialized weights.[/yellow]")
                
            return model
            
        except Exception as e:
            self.console.print(f"[red]Error building U-Net: {e}[/red]")
            return None
            
    def _conv_block(self, input_tensor, num_filters):
        """Convolution block with batch normalization"""
        x = tf.keras.layers.Conv2D(
            num_filters, 
            (3, 3), 
            padding='same',
            kernel_initializer='he_normal'
        )(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        x = tf.keras.layers.Conv2D(
            num_filters, 
            (3, 3), 
            padding='same',
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x
        
    def apply_denoise_methods(self, image):
        """Apply various denoising methods"""
        results = {}
        results['original'] = image.copy()
        
        try:
            # Traditional methods
            results['gaussian'] = cv2.GaussianBlur(image, (5, 5), 1.0)
            results['median'] = cv2.medianBlur(image, 5)
            results['bilateral'] = cv2.bilateralFilter(image, 9, 75, 75)
            
            # U-Net denoising
            unet_input = image.astype(np.float32) / 255.0
            if len(unet_input.shape) == 2:
                unet_input = np.stack([unet_input] * 3, axis=-1)
            elif len(unet_input.shape) == 3 and unet_input.shape[2] == 1:
                unet_input = np.concatenate([unet_input] * 3, axis=-1)
                
            unet_input = np.expand_dims(unet_input, 0)
            denoised = self.unet.predict(unet_input, verbose=0)[0]
            results['unet'] = (denoised * 255).astype(np.uint8)
            
        except Exception as e:
            self.console.print(f"[red]Error in denoising: {e}[/red]")
            
        return results
        
    def apply_sharpen_methods(self, image):
        """Apply various sharpening methods"""
        results = {}
        
        try:
            # Unsharp mask
            gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
            results['unsharp_mask'] = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            # Laplacian
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            results['laplacian'] = cv2.filter2D(image, -1, kernel)
            
        except Exception as e:
            self.console.print(f"[red]Error in sharpening: {e}[/red]")
            
        return results
        
    def compute_metrics(self, image, roi):
        """Compute SNR and edge strength metrics"""
        try:
            x, y, w, h = roi
            roi_img = image[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(roi_img.shape) == 3:
                roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
            else:
                roi_gray = roi_img
                
            # Compute SNR
            signal = np.mean(roi_gray)
            noise = np.std(roi_gray)
            snr = signal / noise if noise > 0 else float('inf')
            
            # Compute edge strength
            sobelx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            
            return snr, edge_strength
            
        except Exception as e:
            self.console.print(f"[red]Error computing metrics: {e}[/red]")
            return 0, 0
            
    def process_image(self, input_path, output_dir):
        """Main processing pipeline"""
        with Progress() as progress:
            task = progress.add_task("Processing image...", total=100)
            
            try:
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Load and preprocess image
                raw_image = self.isp.read_raw(input_path)
                if raw_image is None:
                    return
                    
                progress.update(task, advance=20)
                
                # Apply ISP pipeline
                demosaiced = self.isp.demosaic(raw_image)
                wb_image = self.isp.white_balance(demosaiced)
                base_image = self.isp.apply_gamma(wb_image)
                
                if base_image is None:
                    return
                    
                progress.update(task, advance=20)
                
                # Define ROIs
                rois = [
                    (200, 200, 400, 400),    # Dark region
                    (800, 600, 400, 400),    # Mid-tone region
                    (1400, 800, 400, 400)    # Bright region
                ]
                
                # Apply denoising and sharpening
                denoised_results = self.apply_denoise_methods(base_image)
                progress.update(task, advance=20)
                
                sharpened_results = self.apply_sharpen_methods(base_image)
                progress.update(task, advance=20)
                
                # Compute metrics
                metrics = {}
                for name, img in {**denoised_results, **sharpened_results}.items():
                    metrics[name] = []
                    for i, roi in enumerate(rois):
                        snr, edge_strength = self.compute_metrics(img, roi)
                        metrics[name].append({
                            'region': 'dark' if i == 0 else 'mid' if i == 1 else 'bright',
                            'snr': snr,
                            'edge_strength': edge_strength
                        })
                        
                progress.update(task, advance=10)
                
                # Save results
                self._save_results(denoised_results, sharpened_results, metrics, output_dir, rois)
                progress.update(task, advance=10)
                
                # Display results
                self._display_results(metrics)
                
                return denoised_results, sharpened_results, metrics
                
            except Exception as e:
                self.console.print(f"[red]Fatal error: {e}[/red]")
                return None
                
    def _save_results(self, denoised_results, sharpened_results, metrics, output_dir, rois):
        """Save processed images and metrics"""
        try:
            # Save images
            for name, img in {**denoised_results, **sharpened_results}.items():
                # Save original image
                cv2.imwrite(
                    os.path.join(output_dir, f"{name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
                
                # Save image with ROIs marked
                img_with_roi = img.copy()
                for i, (x, y, w, h) in enumerate(rois):
                    color = [(0,255,0), (255,0,0), (0,0,255)][i]
                    cv2.rectangle(img_with_roi, (x,y), (x+w,y+h), color, 2)
                
                cv2.imwrite(
                    os.path.join(output_dir, f"{name}_roi.png"),
                    cv2.cvtColor(img_with_roi, cv2.COLOR_RGB2BGR)
                )
            
            # Save metrics to CSV
            with open(os.path.join(output_dir, 'metrics.csv'), 'w') as f:
                f.write('Method,Region,SNR,Edge_Strength\n')
                for method, regions in metrics.items():
                    for region in regions:
                        f.write(f"{method},{region['region']},"
                               f"{region['snr']:.2f},{region['edge_strength']:.2f}\n")
                               
        except Exception as e:
            self.console.print(f"[red]Error saving results: {e}[/red]")
            
    def _display_results(self, metrics):
        """Display metrics in a formatted table"""
        table = Table(title="Image Quality Metrics")
        
        table.add_column("Method", justify="left", style="cyan")
        table.add_column("Region", justify="center")
        table.add_column("SNR", justify="right")
        table.add_column("Edge Strength", justify="right")
        
        for method, regions in metrics.items():
            for region in regions:
                table.add_row(
                    method,
                    region['region'],
                    f"{region['snr']:.2f}",
                    f"{region['edge_strength']:.2f}"
                )
                
        self.console.print(table)



def main():
    # Set paths
    input_file = "eSFR_1920x1280_12b_GRGB_6500K_60Lux.raw"
    output_dir = "assignment2_results"
    
    # Initialize pipeline and run
    pipeline = DenoiseSharpenPipeline()
    pipeline.console.print("[yellow]Starting Advanced ISP Pipeline[/yellow]")
    
    results = pipeline.process_image(input_file, output_dir)
    
    pipeline.console.print(f"\n[green]Processing complete! "
                         f"Results saved in: {output_dir}[/green]")

if __name__ == "__main__":
    main()