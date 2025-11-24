# 📷 Bokeh Effect Generator

Interactive web application for generating realistic depth-based bokeh effects using AI-powered depth estimation.

## ✨ Features

- **AI Depth Estimation**: Uses Depth Anything V2 for accurate monocular depth prediction
- **Progressive Blur**: Bucketized depth-based blur for realistic depth-of-field
- **Custom Bokeh Shapes**: Draw your own bokeh kernel shapes (hearts, stars, etc.)
- **Bokeh Splatting**: Realistic bokeh highlights at bright spots with configurable brightness
- **Interactive Focal Plane**: Click to focus or use slider for precise control
- **Depth Visualization**: View depth maps and bucketization analysis
- **Dataset Evaluation**: Calculate PSNR metrics on image datasets

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Linux/macOS (Windows users can use WSL)
- ~4GB disk space for model downloads
- (Optional) CUDA-capable GPU for faster processing

### Installation

```bash
# Clone or navigate to the project directory
# Run the setup script
chmod +x ./setup.sh
./setup.sh
```

### Running the Application

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run the application
python bokeh_app.py
```

The application will start and be available at: **http://localhost:7860**

If you're running on a remote server via SSH, the URL will be displayed in the terminal.

## 📖 Usage Guide

### Basic Workflow

1. **Load Image**
   - Upload an image using the file picker
   - Click "Load & Estimate Depth" to process

2. **Set Focal Depth**
   - Use the slider (0=background, 1=foreground)
   - Or click directly on the image to pick a focus point
   - Green areas = in focus, Red areas = will be blurred

3. **Adjust Bokeh Parameters**
   - **Max Blur**: Controls depth-of-field strength (20-200)
   - **Blur Levels**: Number of progressive blur buckets (5-20)
   - **Layered Bokeh**: Enable for foreground bleed effect
   - **Enable Bokeh Splatting**: Show bokeh shapes at bright spots
   - **Splat Brightness Threshold**: Control which bright spots get bokeh (80-99%)

4. **Generate Bokeh**
   - Click "Generate Bokeh" to create the effect
   - View results in different tabs (Bokeh Result, Comparison, etc.)

### Custom Bokeh Shapes

1. Navigate to "Custom Bokeh Kernel" section
2. Draw a white shape on the black canvas (e.g., heart, star, hexagon)
3. Click "Apply Custom Kernel"
4. Generate bokeh - your custom shape will appear at bright spots!
5. Click "Reset to Gaussian" to return to default circular bokeh

### Dataset Evaluation

1. Place images in `./data/` directory
2. Set desired bokeh parameters
3. Click "Evaluate PSNR on Dataset"
4. Results will show average PSNR across all images


## 📁 Project Structure

```
project/
├── bokeh_app.py          # Main application
├── requirements.txt      # Python dependencies
├── setup.sh             # Automated setup script
├── README.md            # This file
└── data/                # (Optional) Place images here for evaluation
```

## 🔧 Technical Details

### Dependencies

- **torch**: Deep learning framework for model inference
- **transformers**: Hugging Face library for Depth Anything V2 model
- **gradio**: Web UI framework
- **opencv-python**: Image processing
- **scipy**: Signal processing for blur and peak detection
- **numpy**: Numerical operations
- **matplotlib**: Visualization and plotting
- **pillow**: Image I/O

### Model Information

- **Depth Estimation**: [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- **Model Size**: ~100MB download on first run
- **License**: Apache 2.0

### Algorithm Overview

1. **Depth Estimation**: Monocular depth map using Depth Anything V2
2. **Depth Bucketization**: Discretize depth into blur levels
3. **Progressive Blur**: Apply Gaussian blur scaled by distance from focal plane
4. **Peak Detection**: Find local maxima in brightness (scipy maximum_filter)
5. **Bokeh Splatting**: Place bokeh kernels at bright peaks in out-of-focus regions
6. **Layered Composition**: Blend foreground/background with proper occlusion

## 📝 License

This project uses the Depth Anything V2 model under the Apache 2.0 license.

## 🙏 Credits

- **Depth Anything V2**: [LiheYoung/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Gradio**: Web UI framework by Hugging Face


**Enjoy creating beautiful bokeh effects! 🎨📸**
