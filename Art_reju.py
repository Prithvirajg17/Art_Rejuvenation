import gradio as gr
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
import tempfile
import os

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16"
)


device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)


generator = torch.manual_seed(92)
if device == "cuda":
    torch.cuda.manual_seed_all(92)

def inpaint(init_image, mask_image, prompt):
    try:
        result_image = pipeline(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator
        ).images[0]
        return result_image, None
    except Exception as e:
        return None, f"An error occurred during inpainting: {str(e)}"

def save_image(image):
    """Save the image to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file, format="PNG")
        return temp_file.name

def main(init_image, mask_image, prompt):
    try:
        init_image = Image.open(init_image).convert("RGB")
        mask_image = Image.open(mask_image).convert("RGB")
    except Exception as e:
        return None, None, f"Error loading images: {str(e)}"

    result, error_message = inpaint(init_image, mask_image, prompt)

    if error_message:
        return None, None, error_message

    if not isinstance(result, Image.Image):
        return None, None, "Inpainting did not return a valid image."

    return save_image(init_image), save_image(mask_image), save_image(result)

with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .title {
        text-align: center;
        font-size: 60px; 
        font-family: "Bree Serif", serif; 
        color: #2cff05; 
        margin-bottom: 20px; 
        font-weight: bold;
    }
"""
) as demo:
    gr.Markdown("<div class='title'>Art Rejuvenation</div>")
    gr.Markdown("Upload an initial image and a mask image, then enter a prompt for Reconstructing.")

    with gr.Row():
        init_image_input = gr.Image(label="Initial Image", type="filepath")
        mask_image_input = gr.Image(label="Mask Image", type="filepath")

    prompt_input = gr.Textbox(
        label="Prompt",
        placeholder="Enter a prompt for inpainting...",
        value="concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
    )

    with gr.Row():
        inpaint_button = gr.Button("Refine", elem_classes=["custom-button"])

    with gr.Row():
        init_image_output = gr.Image(label="Original Image", width=300, height=400)
        mask_image_output = gr.Image(label="Mask Image", width=300, height=400)
        result_output = gr.Image(label="Reconstructed Image", width=300, height=400)

    inpaint_button.click(
        main,
        inputs=[init_image_input, mask_image_input, prompt_input],
        outputs=[init_image_output, mask_image_output, result_output]
    )


demo.launch(share=True)
