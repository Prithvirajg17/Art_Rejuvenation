import gradio as gr
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
import tempfile
import os
from cloudant.client import Cloudant
from cloudant.error import CloudantException
from io import BytesIO
import base64

# Configure the Cloudant client
CLOUDANT_USERNAME = "f8b366e8-91d7-4875-8f50-c352ccb8a688-bluemix"
CLOUDANT_API_KEY = "EXQ_7-gxRJa2rm57fK4IPenIs1MHhAOJaCsuTbLNr3J3"
CLOUDANT_URL = "https://f8b366e8-91d7-4875-8f50-c352ccb8a688-bluemix.cloudantnosqldb.appdomain.cloud"

client = Cloudant.iam(CLOUDANT_USERNAME, CLOUDANT_API_KEY, connect=True)
db_name = "reconstruct"
if db_name not in client.all_dbs():
    client.create_database(db_name)
db = client[db_name]

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

def save_to_cloudant(image, prompt):
    """Save the reconstructed image and prompt to IBM Cloudant."""
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Create a document with the image and prompt
        doc = {
            "prompt": prompt,
            "image": {
                "data": image_data,
                "format": "png"
            }
        }
        db.create_document(doc)
        return "Image successfully stored in Cloudant."
    except Exception as e:
        return f"Failed to store image in Cloudant: {str(e)}"

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

    # Save to Cloudant
    cloudant_message = save_to_cloudant(result, prompt)

    return save_image(init_image), save_image(mask_image), save_image(result), cloudant_message

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
        cloudant_status = gr.Textbox(label="Cloudant Status", interactive=False)

    inpaint_button.click(
        main,
        inputs=[init_image_input, mask_image_input, prompt_input],
        outputs=[init_image_output, mask_image_output, result_output, cloudant_status]
    )

demo.launch(share=True)
