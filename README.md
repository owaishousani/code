Main Features:
1. Cross-Modal Attention Extraction

Extracts attention weights between text tokens and image patch tokens
Works with Qwen2-VL models that process both text and images
Identifies image token positions and text token positions in the sequence

2. Spatial Attention Heatmaps

create_attention_heatmap_on_image(): Creates attention heatmaps overlaid on the original image
Shows which image regions the model focuses on when processing specific text tokens
Supports layer-specific and head-specific analysis

3. Comprehensive Visualization

visualize_cross_modal_attention(): Creates multi-layer visualization showing:

Original image
Attention heatmaps for each layer
Attention overlaid on the image


Shows how attention patterns evolve across different layers

4. Detailed Attention Matrix

create_token_image_attention_matrix(): Creates a detailed heatmap showing attention between individual text tokens and image patches
Helps identify which specific words attend to which image regions

Key Capabilities:
Text-to-Image Attention
python# Analyze where "red cat" tokens look in the image
visualizer.visualize_cross_modal_attention(
    image="path/to/image.jpg",
    text_prompt="Where is the red cat on the image?",
    target_phrase="red cat",
    layers_to_show=[-4, -3, -2, -1]  # Last 4 layers
)
Token-Level Analysis
python# See attention from each text token to image regions
attention_matrix = visualizer.create_token_image_attention_matrix(
    image="path/to/image.jpg",
    text_prompt="Where is the red cat on the image?",
    layer_idx=-1  # Last layer
)
How It Works:

Input Processing: The model processes both image and text together
Attention Extraction: Extracts cross-attention weights between text tokens and image patch tokens
Spatial Mapping: Maps image token attention back to spatial coordinates in the original image
Visualization: Creates heatmaps showing attention intensity across image regions

Usage Example:
python# Initialize visualizer
visualizer = QwenVLCrossModalVisualizer("Qwen/Qwen2-VL-7B-Instruct")

# Analyze cross-modal attention
visualizer.visualize_cross_modal_attention(
    image="your_image.jpg",
    text_prompt="Where is the red cat on the image?",
    target_phrase="red cat",
    save_path="attention_analysis.png"
)
Installation Requirements:
bashpip install torch transformers pillow opencv-python matplotlib seaborn numpy
Key Differences from Basic Attention Maps:

Cross-Modal: Shows text↔image connections, not just text↔text
Spatial: Maps attention back to actual image coordinates
Multi-Layer: Shows how cross-modal attention evolves through layers
Token-Specific: Focuses on specific phrases like "red cat"
Visual: Overlays attention on the actual image for intuitive understanding

This tool will help you understand exactly which parts of an image the model is "looking at" when processing specific text tokens, providing insights into the model's visual reasoning process.
