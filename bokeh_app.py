# Core ML and image processing libraries
import torch
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
from transformers import pipeline  # For Depth Anything V2 model
import gradio as gr  # Web UI framework
import matplotlib.pyplot as plt
import io
import os
import glob


class BokehGenerator:
    """Generates realistic bokeh (depth-of-field blur) effects using depth estimation."""
    
    def __init__(self, device=None):
        """Initialize bokeh generator with optional device specification.
        
        Args:
            device: Device to run depth estimation on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_estimator = None  # Lazy-loaded to save memory
        
    def _load_depth_estimator(self):
        """Lazy-load Depth Anything V2 model to avoid loading until needed."""
        if self.depth_estimator is None:
            print("Loading Depth Anything V2 model...")
            self.depth_estimator = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf"
            )
    
    def estimate_depth(self, image):
        """Estimate depth map from RGB image using Depth Anything V2.
        
        Args:
            image: PIL Image object
            
        Returns:
            Normalized depth map (0=background, 1=foreground) as numpy array
        """
        self._load_depth_estimator()
        
        # Get depth prediction from pretrained model
        depth_result = self.depth_estimator(image)
        depth_map = depth_result["depth"]
        
        # Convert to numpy array and resize to match input image dimensions
        depth_array = np.array(depth_map)
        depth_array = cv2.resize(
            depth_array,
            (image.size[0], image.size[1]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to 0-1 range (0=far/background, 1=near/foreground)
        depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        
        return depth_normalized
    
    def apply_bokeh(self, image_np, depth_normalized, focal_depth=None,
                   num_buckets=10, max_blur_kernel=100, layered=True, 
                   custom_kernel=None, use_splatting=True, splat_percentile=95):
        """Apply bokeh effect to image based on depth map.
        
        Args:
            image_np: Input image as numpy array (H, W, 3)
            depth_normalized: Normalized depth map (0-1)
            focal_depth: Depth value to keep in focus (0-1). None defaults to 0.5
            num_buckets: Number of discrete blur levels to use
            max_blur_kernel: Maximum blur kernel size for most out-of-focus areas
            layered: If True, apply layered rendering (foreground bleeds over background)
            custom_kernel: Optional custom bokeh kernel shape (2D array)
            use_splatting: Enable bright point splatting for realistic bokeh highlights
            splat_percentile: Brightness percentile threshold for splatting (higher = fewer splats)
            
        Returns:
            Tuple of (bokeh_image, bucket_map, bucket_edges)
        """
        
        def apply_bokeh_blur_with_splat(img, kernel_size, bokeh_kernel=None, enable_splat=True, percentile=95):
            """Apply Gaussian blur with optional bright point splatting for bokeh highlights.
            
            Args:
                img: Input image patch
                kernel_size: Size of blur kernel
                bokeh_kernel: Optional custom kernel shape
                enable_splat: Whether to apply splatting effect
                percentile: Brightness percentile for detecting highlights
                
            Returns:
                Blurred image with bokeh splatting applied
            """
            # Skip blur for sharp regions (kernel size = 1)
            if kernel_size <= 1:
                return img.astype(np.float32)
            
            # Convert kernel size to Gaussian sigma (rule of thumb: kernel ≈ 6σ)
            sigma = kernel_size / 6.0
            
            # Start with base Gaussian blur applied per color channel
            blurred = np.zeros_like(img, dtype=np.float32)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(
                    img[:, :, c].astype(np.float32),
                    sigma=sigma,
                    mode='reflect'
                )
            
            # Apply bokeh splatting for bright highlights (creates circular bokeh balls)
            if enable_splat and kernel_size > 10:
                # Determine splat kernel size (must be odd)
                bokeh_size = max(15, min(101, kernel_size))
                if bokeh_size % 2 == 0:
                    bokeh_size += 1
                
                # Convert to grayscale to detect bright points
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
                
                # Find local maxima (bright points) using maximum filter
                local_max = maximum_filter(gray, size=max(3, bokeh_size // 3))
                peak_mask = (gray == local_max) & (gray > np.percentile(gray, percentile))
                
                # Get peak locations
                # Get coordinates of all bright points to splat
                peak_coords = np.argwhere(peak_mask)
                
                if len(peak_coords) > 0:
                    # Use custom kernel if provided, otherwise create circular kernel
                    if bokeh_kernel is not None:
                        # Resize custom kernel to appropriate size
                        kernel_resized = cv2.resize(bokeh_kernel, (bokeh_size, bokeh_size), 
                                                   interpolation=cv2.INTER_LINEAR)
                        # Normalize to sum to 1 (preserves brightness)
                        if kernel_resized.sum() > 0:
                            kernel_resized = kernel_resized / kernel_resized.sum()
                        splat_kernel = kernel_resized
                    else:
                        # Create default circular bokeh kernel
                        ax = np.arange(-bokeh_size // 2 + 1., bokeh_size // 2 + 1.)
                        xx, yy = np.meshgrid(ax, ax)
                        radius = bokeh_size / 2.2
                        splat_kernel = (xx**2 + yy**2 <= radius**2).astype(np.float32)
                        if splat_kernel.sum() > 0:
                            splat_kernel = splat_kernel / splat_kernel.sum()
                    
                    # Splat each bright point with the bokeh kernel
                    for py, px in peak_coords:
                        # Scale splat intensity by pixel brightness
                        brightness = gray[py, px] / 255.0
                        
                        # Calculate image patch coordinates (with bounds checking)
                        half_size = bokeh_size // 2
                        y_start = max(0, py - half_size)
                        y_end = min(img.shape[0], py + half_size + 1)
                        x_start = max(0, px - half_size)
                        x_end = min(img.shape[1], px + half_size + 1)
                        
                        # Calculate corresponding kernel patch coordinates
                        ky_start = half_size - (py - y_start)
                        ky_end = ky_start + (y_end - y_start)
                        kx_start = half_size - (px - x_start)
                        kx_end = kx_start + (x_end - x_start)
                        
                        # Extract relevant kernel patch (handles edge cases)
                        kernel_patch = splat_kernel[ky_start:ky_end, kx_start:kx_end]
                        
                        # Apply splat to each color channel with brightness boost
                        for c in range(3):
                            color = img[py, px, c] * brightness * 8.0  # Strong boost for visible bokeh balls
                            blurred[y_start:y_end, x_start:x_end, c] += color * kernel_patch
            
            return np.clip(blurred, 0, 255)

        h, w = image_np.shape[:2]
        
        # Default to middle depth if not specified
        if focal_depth is None:
            focal_depth = 0.5
        
        # Calculate distance from focal plane (0 = in focus)
        depth_distance = np.abs(depth_normalized - focal_depth)
        
        # Create discrete blur levels (buckets) based on depth distance
        max_distance = depth_distance.max()
        bucket_edges = np.linspace(0, max_distance, num_buckets + 1)
        
        # Assign progressively larger blur kernels to each bucket
        blur_kernels = [1 + int(i * (max_blur_kernel / num_buckets)) for i in range(num_buckets)]
        blur_kernels = [k if k % 2 == 1 else k + 1 for k in blur_kernels]  # Ensure odd (required for kernels)
        
        # Assign each pixel to a blur bucket
        bucket_map = np.digitize(depth_distance, bucket_edges) - 1
        bucket_map = np.clip(bucket_map, 0, num_buckets - 1)
        
        # Two rendering modes: simple (non-layered) vs. layered
        if not layered:
            # Simple mode: Apply appropriate blur to each pixel based on its bucket
            bokeh_image = np.zeros_like(image_np, dtype=np.float32)
            
            for bucket_idx in range(num_buckets):
                bucket_mask = (bucket_map == bucket_idx)
                if bucket_mask.sum() == 0:
                    continue
                
                kernel_size = blur_kernels[bucket_idx]
                
                if kernel_size == 1:
                    bokeh_image[bucket_mask] = image_np[bucket_mask]
                else:
                    blurred = apply_bokeh_blur_with_splat(image_np, kernel_size, custom_kernel, use_splatting, splat_percentile)
                    bokeh_image[bucket_mask] = blurred[bucket_mask]
        else:
            # Layered mode: Render background then foreground (more realistic depth layering)
            # Separate pixels into foreground (closer) and background (farther)
            foreground_blur_mask = depth_normalized > focal_depth
            background_blur_mask = depth_normalized <= focal_depth
            
            bokeh_image = image_np.astype(np.float32).copy()
            
            # Pre-compute all blur levels to avoid redundant computation
            blurred_images = []
            for bucket_idx in range(num_buckets):
                kernel_size = blur_kernels[bucket_idx]
                if kernel_size == 1:
                    blurred_images.append(image_np.astype(np.float32))
                else:
                    blurred = apply_bokeh_blur_with_splat(image_np, kernel_size, custom_kernel, use_splatting, splat_percentile)
                    blurred_images.append(blurred)
            
            # First pass: Apply blur to background pixels (layer 0)
            for bucket_idx in range(num_buckets):
                bucket_mask = (bucket_map == bucket_idx) & background_blur_mask
                if bucket_mask.sum() == 0:
                    continue
                bokeh_image[bucket_mask] = blurred_images[bucket_idx][bucket_mask]
            
            # Second pass: Blend foreground pixels on top (creates realistic foreground bleed)
            for bucket_idx in range(num_buckets):
                bucket_mask = (bucket_map == bucket_idx) & foreground_blur_mask
                if bucket_mask.sum() == 0:
                    continue
                
                # Blend foreground blur with existing background (alpha compositing)
                alpha = 0.7
                bokeh_image[bucket_mask] = (
                    alpha * blurred_images[bucket_idx][bucket_mask] +
                    (1 - alpha) * bokeh_image[bucket_mask]
                )
        
        return bokeh_image.astype(np.uint8), bucket_map, bucket_edges


class BokehApp:
    """Gradio web application for interactive bokeh generation."""
    
    def __init__(self):
        """Initialize the bokeh app with default state."""
        self.generator = BokehGenerator()
        self.depth_map = None  # Cached depth estimation
        self.image_np = None  # Current input image
        self.focal_depth = 0.5  # Current focal depth setting
        self.custom_kernel = None  # User-drawn bokeh kernel
        
    def process_image(self, image):
        """Load and process an input image, estimating its depth map.
        
        Args:
            image: PIL Image from Gradio upload
            
        Returns:
            Tuple of (original_image, depth_vis, focal_vis, focal_depth, status_message)
        """
        """Load and process an input image, estimating its depth map.
        
        Args:
            image: PIL Image from Gradio upload
            
        Returns:
            Tuple of (original_image, depth_vis, focal_vis, focal_depth, status_message)
        """
        if image is None:
            return None, None, None, 0.5, "Please upload an image first"
        
        # Convert to numpy array for processing
        self.image_np = np.array(image)
        
        # Run depth estimation (this may take a few seconds)
        self.depth_map = self.generator.estimate_depth(image)
        
        # Reset focal depth to middle
        self.focal_depth = 0.5
        
        # Create colored visualization of depth map (using plasma colormap)
        depth_vis = (self.depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        focal_vis = self.visualize_focal_plane(0.5)
        
        return image, Image.fromarray(depth_colored), focal_vis, 0.5, "Image loaded and depth estimated!"
    
    def update_focal_depth(self, focal_depth):
        """Update focal depth slider and regenerate focal plane visualization.
        
        Args:
            focal_depth: New focal depth value (0-1)
            
        Returns:
            Tuple of (focal_visualization, status_message)
        """
        if self.image_np is None or self.depth_map is None:
            return None, "Load an image first!"
        
        self.focal_depth = focal_depth
        focal_vis = self.visualize_focal_plane(focal_depth)
        
        return focal_vis, f"Focal depth: {focal_depth:.3f}"
    
    def pick_focal_depth_from_click(self, image_with_point, evt: gr.SelectData):
        """Allow user to click on image to set focal depth at that point.
        
        Args:
            image_with_point: The clicked image (not used)
            evt: Gradio SelectData event containing click coordinates
            
        Returns:
            Tuple of (new_focal_depth, focal_visualization, status_message)
        """
        if self.depth_map is None or evt is None:
            return 0.5, None, "Load an image first!"
        
        # Get click coordinates from Gradio event
        x, y = evt.index
        
        # Sample depth value at clicked location
        h, w = self.depth_map.shape
        if 0 <= y < h and 0 <= x < w:
            focal_depth = float(self.depth_map[y, x])
            self.focal_depth = focal_depth
            
            focal_vis = self.visualize_focal_plane(focal_depth)
            return focal_depth, focal_vis, f"Focal depth picked: {focal_depth:.3f} at ({x}, {y})"
        
        return self.focal_depth, None, "Click inside the image"
    
    def visualize_focal_plane(self, focal_depth):
        """Create color-coded visualization showing which areas are in/out of focus.
        
        Green = in focus, Red = out of focus
        
        Args:
            focal_depth: Current focal depth value (0-1)
            
        Returns:
            PIL Image with colored overlay
        """
        if self.image_np is None or self.depth_map is None:
            return None
        
        # Calculate distance from focal plane
        depth_distance = np.abs(self.depth_map - focal_depth)
        
        # Normalize to 0-1 range
        if depth_distance.max() > 0:
            depth_distance_norm = depth_distance / depth_distance.max()
        else:
            depth_distance_norm = depth_distance
        
        overlay = self.image_np.copy()
        focus_map = (1 - depth_distance_norm)  # Inverted: 1 = in focus
        
        # Create RGB overlay: Green for focus, Red for blur
        colored_overlay = np.zeros_like(self.image_np)
        colored_overlay[:, :, 1] = (focus_map * 255).astype(np.uint8)  # Green channel = focus
        colored_overlay[:, :, 0] = (depth_distance_norm * 255).astype(np.uint8)  # Red channel = blur
        
        # Blend overlay with original image
        overlay = cv2.addWeighted(overlay, 0.7, colored_overlay, 0.3, 0)
        
        return Image.fromarray(overlay)
    
    def process_kernel_drawing(self, kernel_image):
        """Process user-drawn bokeh kernel from Gradio Paint component.
        
        Converts user drawing into normalized kernel for custom bokeh shapes.
        
        Args:
            kernel_image: Drawing from Gradio Paint component (dict or PIL Image)
            
        Returns:
            Tuple of (kernel_preview_image, status_message)
        """
        if kernel_image is None:
            self.custom_kernel = None
            return None, "No kernel drawn"
        
        try:
            # Handle Gradio Paint component output (can be dict with layers)
            if isinstance(kernel_image, dict):
                if 'composite' in kernel_image and kernel_image['composite'] is not None:
                    kernel_pil = kernel_image['composite']
                elif 'layers' in kernel_image and len(kernel_image['layers']) > 0:
                    kernel_pil = kernel_image['layers'][0]
                elif 'background' in kernel_image and kernel_image['background'] is not None:
                    kernel_pil = kernel_image['background']
                else:
                    return None, "Could not extract drawing from Paint component"
            else:
                kernel_pil = kernel_image
            
            kernel_np = np.array(kernel_pil)
            
            # Convert to grayscale, handling RGBA/RGB/grayscale inputs
            if len(kernel_np.shape) == 3:
                if kernel_np.shape[2] == 4:  # RGBA
                    alpha = kernel_np[:, :, 3]
                    rgb = kernel_np[:, :, :3]
                    kernel_gray = np.mean(rgb, axis=2) * (alpha / 255.0)
                else:  # RGB
                    kernel_gray = np.mean(kernel_np, axis=2)
            else:  # Already grayscale
                kernel_gray = kernel_np
            
            # Threshold to binary (white drawing on black = kernel shape)
            _, kernel_binary = cv2.threshold(kernel_gray.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)
            kernel_binary = 255 - kernel_binary  # Invert: now white pixels = kernel
            
            # Normalize to 0-1 range
            kernel_normalized = kernel_binary.astype(np.float32) / 255.0
            
            if kernel_normalized.sum() == 0:
                return None, "No drawing detected. Draw a white shape!"
            
            # Find bounding box of drawn shape
            rows = np.any(kernel_normalized > 0, axis=1)
            cols = np.any(kernel_normalized > 0, axis=0)
            
            if rows.sum() > 0 and cols.sum() > 0:
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                # Add small padding around shape
                pad = 5
                rmin = max(0, rmin - pad)
                rmax = min(kernel_normalized.shape[0], rmax + pad + 1)
                cmin = max(0, cmin - pad)
                cmax = min(kernel_normalized.shape[1], cmax + pad + 1)
                
                # Crop to bounding box
                kernel_cropped = kernel_normalized[rmin:rmax, cmin:cmax]
            else:
                return None, "No drawing detected!"
            
            # Make kernel square (required for consistent rotation)
            h, w = kernel_cropped.shape
            max_dim = max(h, w)
            kernel_square = np.zeros((max_dim, max_dim), dtype=np.float32)
            
            # Center the cropped kernel in square canvas
            h_offset = (max_dim - h) // 2
            w_offset = (max_dim - w) // 2
            kernel_square[h_offset:h_offset+h, w_offset:w_offset+w] = kernel_cropped
            
            # Normalize kernel to sum to 1 (preserves energy/brightness)
            if kernel_square.sum() > 0:
                kernel_square = kernel_square / kernel_square.sum()
            else:
                return None, "Kernel is empty!"
            
            # Store processed kernel for bokeh generation
            self.custom_kernel = kernel_square
            
            kernel_vis = (kernel_square / kernel_square.max() * 255).astype(np.uint8) if kernel_square.max() > 0 else kernel_square.astype(np.uint8)
            display_size = 200
            kernel_vis_large = cv2.resize(kernel_vis, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            kernel_vis_rgb = cv2.cvtColor(kernel_vis_large, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(kernel_vis_rgb), f"✓ Custom kernel loaded! Shape: {kernel_square.shape}, Non-zero pixels: {(kernel_square > 0).sum()}"
        except Exception as e:
            import traceback
            return None, f"Error processing kernel: {str(e)}\n{traceback.format_exc()}"
    
    def reset_kernel(self):
        """Reset to default Gaussian bokeh (circular)."""
        self.custom_kernel = None
        return None, None, "Reset to Gaussian bokeh"
    
    def generate_bokeh(self, focal_depth, max_blur, num_buckets, use_layered, use_splatting, splat_threshold):
        """Generate final bokeh image with current settings.
        
        Args:
            focal_depth: Depth value to keep in focus (0-1)
            max_blur: Maximum blur kernel size
            num_buckets: Number of discrete blur levels
            use_layered: Enable layered rendering
            use_splatting: Enable bright point splatting
            splat_threshold: Brightness percentile for splatting
            
        Returns:
            Tuple of (bokeh_image, comparison_image, bucket_visualization, status_message)
        """
        if self.image_np is None or self.depth_map is None:
            return None, None, None, "Load an image first!"
        
        self.focal_depth = focal_depth
        
        # Generate bokeh effect using current parameters
        bokeh_image, bucket_map, bucket_edges = self.generator.apply_bokeh(
            self.image_np,
            self.depth_map,
            focal_depth=focal_depth,
            num_buckets=int(num_buckets),
            max_blur_kernel=int(max_blur),
            layered=use_layered,
            custom_kernel=self.custom_kernel,
            use_splatting=use_splatting,
            splat_percentile=splat_threshold
        )
        
        # Prepare status message
        kernel_msg = " (Custom kernel)" if self.custom_kernel is not None else " (Gaussian blur)"
        
        # Create side-by-side comparison
        comparison = np.hstack([self.image_np, bokeh_image])
        
        # Generate visualization of blur bucketization
        bucket_plot = self.visualize_buckets(bucket_map, bucket_edges, int(num_buckets))
        
        return Image.fromarray(bokeh_image), Image.fromarray(comparison), bucket_plot, f"Bokeh generated! Focal depth: {focal_depth:.3f}{kernel_msg}"
    
    def visualize_buckets(self, bucket_map, bucket_edges, num_buckets):
        """Create visualization showing how image is divided into blur buckets.
        
        Args:
            bucket_map: 2D array mapping pixels to bucket indices
            bucket_edges: Bucket boundary values
            num_buckets: Total number of buckets
            
        Returns:
            PIL Image with bucket visualization (heatmap + histogram)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left: Spatial bucket map (heatmap showing which pixels get which blur level)
        im = ax1.imshow(bucket_map, cmap='jet', vmin=0, vmax=num_buckets-1)
        ax1.set_title('Depth Bucketization Map')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, label='Bucket Index', fraction=0.046, pad=0.04)
        
        # Right: Histogram showing pixel distribution across buckets
        bucket_counts = [np.sum(bucket_map == i) for i in range(num_buckets)]
        ax2.bar(range(num_buckets), bucket_counts, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Bucket Index (0=in focus, higher=more blur)')
        ax2.set_ylabel('Pixel Count')
        ax2.set_title('Pixels per Blur Bucket')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf)
    
    def evaluate_psnr_on_dataset(self, focal_depth, max_blur, num_buckets, use_layered, use_splatting, splat_threshold, progress=gr.Progress()):
        """Evaluate bokeh quality using PSNR on dataset in ./data/ directory.
        
        Note: PSNR measures difference from original (higher = more similar).
        This is mainly for debugging/comparison purposes.
        
        Args:
            focal_depth, max_blur, num_buckets, use_layered, use_splatting, splat_threshold: Bokeh parameters
            progress: Gradio Progress tracker
            
        Returns:
            Formatted string with PSNR results for each image
        """
        data_dir = "./data"
        image_files = glob.glob(os.path.join(data_dir, "*.jpg"))
        
        if len(image_files) == 0:
            return "No images found in ./data/ directory"
        
        total_psnr = 0.0
        num_images = len(image_files)
        results = []
        
        progress(0, desc="Starting PSNR evaluation...")
        
        # Process each image in dataset
        for i, img_path in enumerate(image_files):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                image_np = np.array(image)
                
                # Estimate depth for this image
                depth_map = self.generator.estimate_depth(image)
                
                # Generate bokeh with current parameters
                bokeh_image, _, _ = self.generator.apply_bokeh(
                    image_np,
                    depth_map,
                    focal_depth=focal_depth,
                    num_buckets=int(num_buckets),
                    max_blur_kernel=int(max_blur),
                    layered=use_layered,
                    custom_kernel=None,
                    use_splatting=use_splatting,
                    splat_percentile=splat_threshold
                )
                
                # Calculate PSNR (Peak Signal-to-Noise Ratio)
                # Measures similarity to original: higher = more similar (less blur applied)
                mse = np.mean((image_np.astype(np.float32) - bokeh_image.astype(np.float32)) ** 2)
                if mse == 0:
                    psnr = 100  # Perfect match
                else:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                
                total_psnr += psnr
                results.append(f"{os.path.basename(img_path)}: {psnr:.2f} dB")
                
                progress((i + 1) / num_images, desc=f"Processed {i + 1}/{num_images} images")
                
            except Exception as e:
                results.append(f"{os.path.basename(img_path)}: ERROR - {str(e)}")
        
        avg_psnr = total_psnr / num_images
        
        summary = f"📊 PSNR Evaluation Results\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"Total Images: {num_images}\n"
        summary += f"Average PSNR: {avg_psnr:.2f} dB\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        summary += "Individual Results:\n"
        summary += "\n".join(results[:20])
        if len(results) > 20:
            summary += f"\n... and {len(results) - 20} more images"
        
        return summary


def create_ui():
    """Create and configure the Gradio web interface.
    
    Returns:
        Gradio Blocks demo object ready to launch
    """
    app = BokehApp()
    
    with gr.Blocks(title="Bokeh Effect Generator") as demo:
        gr.Markdown("# Bokeh Effect Generator")
        gr.Markdown("Generate realistic depth-based bokeh effects with customizable blur kernels")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Load Image")
                input_image = gr.Image(type="pil", label="Upload Image")
                load_btn = gr.Button("🔍 Load & Estimate Depth", variant="primary")
                
                gr.Markdown("### Set Focal Depth")
                focal_depth_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label="Focal Depth (0=background, 1=foreground)",
                    info="Click on image to pick depth"
                )
                click_image = gr.Image(type="pil", label="Click to Pick Focal Point")
                
                gr.Markdown("### Custom Bokeh Kernel (Optional)")
                gr.Markdown("**Draw a white shape on black background**")
                
                kernel_canvas = gr.Paint(
                    label="Draw Bokeh Kernel Shape",
                    brush=gr.Brush(colors=["black"], default_size=40),
                    height=300,
                    width=300
                )
                
                with gr.Row():
                    apply_kernel_btn = gr.Button("Apply Custom Kernel", size="sm", variant="primary")
                    reset_kernel_btn = gr.Button("Reset to Gaussian", size="sm")
                
                kernel_preview = gr.Image(label="Kernel Preview", height=200)
                
                gr.Markdown("### Bokeh Parameters")
                
                max_blur = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=70,
                    step=5,
                    label="Max Blur (Depth of Field Strength)"
                )
                
                num_buckets = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=16,
                    step=1,
                    label="Blur Levels"
                )
                
                use_layered = gr.Checkbox(
                    value=True,
                    label="Use Layered Bokeh (foreground bleed)"
                )
                
                use_splatting = gr.Checkbox(
                    value=True,
                    label="Enable Bokeh Splatting"
                )
                
                splat_threshold = gr.Slider(
                    minimum=80,
                    maximum=99,
                    value=98,
                    step=1,
                    label="Splat Brightness Threshold (%)"
                )
                
                generate_btn = gr.Button("🎨 Generate Bokeh", variant="primary", size="lg")
                
                status_text = gr.Textbox(label="Status", value="Upload an image to begin", interactive=False)
                
                gr.Markdown("### 📊 Dataset Evaluation")
                eval_btn = gr.Button("📈 Evaluate PSNR on Dataset (./data/*.jpg)", variant="secondary", size="sm")
                eval_output = gr.Textbox(label="Evaluation Results", lines=15, max_lines=30)
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Original"):
                        original_output = gr.Image(label="Original Image")
                    
                    with gr.Tab("Depth Map"):
                        depth_output = gr.Image(label="Estimated Depth")
                    
                    with gr.Tab("Focal Plane Visualization"):
                        focal_vis_output = gr.Image(label="Focal Plane (Green=Focus, Red=Blur)")
                    
                    with gr.Tab("Custom Kernel"):
                        gr.Markdown("### Your custom bokeh kernel")
                        kernel_display = gr.Image(label="Active Bokeh Kernel Shape")
                    
                    with gr.Tab("Bokeh Result"):
                        bokeh_output = gr.Image(label="Bokeh Effect")
                    
                    with gr.Tab("Comparison"):
                        comparison_output = gr.Image(label="Before | After")
                    
                    with gr.Tab("Depth Bucketization"):
                        bucket_vis_output = gr.Image(label="Bucket Map & Distribution")
        
        # Event handlers - wire up UI components to callback functions
        
        # Load button: process image and estimate depth
        load_btn.click(
            fn=app.process_image,
            inputs=[input_image],
            outputs=[original_output, depth_output, focal_vis_output, focal_depth_slider, status_text]
        ).then(
            # Copy image to click canvas
            fn=lambda img: img,
            inputs=[input_image],
            outputs=[click_image]
        )
        
        # Focal depth slider: update visualization when slider moves
        focal_depth_slider.change(
            fn=app.update_focal_depth,
            inputs=[focal_depth_slider],
            outputs=[focal_vis_output, status_text]
        )
        
        # Click-to-focus: set focal depth by clicking on image
        click_image.select(
            fn=app.pick_focal_depth_from_click,
            inputs=[click_image],
            outputs=[focal_depth_slider, focal_vis_output, status_text]
        )
        
        # Custom kernel buttons
        apply_kernel_btn.click(
            fn=app.process_kernel_drawing,
            inputs=[kernel_canvas],
            outputs=[kernel_preview, status_text]
        ).then(
            # Copy preview to display tab
            fn=lambda x: x,
            inputs=[kernel_preview],
            outputs=[kernel_display]
        )
        
        reset_kernel_btn.click(
            fn=app.reset_kernel,
            inputs=[],
            outputs=[kernel_preview, kernel_display, status_text]
        )
        
        # Generate bokeh button: main action
        generate_btn.click(
            fn=app.generate_bokeh,
            inputs=[focal_depth_slider, max_blur, num_buckets, use_layered, use_splatting, splat_threshold],
            outputs=[bokeh_output, comparison_output, bucket_vis_output, status_text]
        )
        
        # Dataset evaluation button
        eval_btn.click(
            fn=app.evaluate_psnr_on_dataset,
            inputs=[focal_depth_slider, max_blur, num_buckets, use_layered, use_splatting, splat_threshold],
            outputs=[eval_output]
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the Gradio web app
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create public URL
        show_error=True  # Display errors in UI
    )
