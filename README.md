# Art_Rejuvenation

Art Rejuvenation is a web-based application that allows users to perform image Restoration using a Stable Diffusion model. Users can upload an initial image(broken old statue or distorted paintings image) and a mask image(making of Distorted part) , provide a textual prompt, and obtain a reconstructed image based on the prompt. The reconstructed image is also stored securely in an IBM Cloudant database.

---

## Features

1. **Image Inpainting**:
   - Uses a pretrained Stable Diffusion model for high-quality image inpainting.

2. **Cloud Storage**:
   - Saves the reconstructed image and the associated prompt to IBM Cloudant for future access or analysis.

3. **Interactive Interface**:
   - Built with Gradio to provide an intuitive and user-friendly experience.

4. **Customization**:
   - Users can specify detailed prompts for inpainting to achieve personalized results.

---

## Technologies Used

### Libraries
- **Gradio**: For creating the interactive user interface.
- **PyTorch**: For leveraging GPU-accelerated deep learning computations.
- **Diffusers**: For inpainting using the Stable Diffusion model.
- **Pillow (PIL)**: For image manipulation.
- **IBM Cloudant**: For storing reconstructed images and prompts securely in the cloud.
- **Base64 and BytesIO**: For encoding image data to store in Cloudant.
- **Tempfile**: For creating temporary files during image processing.

### Model
- **Stable Diffusion**: A pretrained model for image generation and inpainting provided by the `diffusers` library.

---

## Prerequisites

1. Python 3.7+
2. `pip` for package installation
3. IBM Cloudant account with access credentials
4. NVIDIA GPU with CUDA (optional for better performance)

---

## Usage

1. **Launch the Application**:
   ```bash
   python Art_rejuvenation.py
   ```

2. **Access the Interface**:
   - Open the provided URL in your web browser (e.g., `http://127.0.0.1:7860`).

3. **Perform Inpainting**:
   - Upload an initial image and a mask image.
   - Enter a textual prompt describing the desired reconstruction.
   - Click on the "Refine" button.

4. **View Results**:
   - See the original, mask, and reconstructed images in the interface.
   - Check the Cloudant status to confirm if the image is stored successfully.

---

## Code Walkthrough

### Key Components

1. **Cloudant Integration**:
   - Configure the Cloudant client using `Cloudant.iam()`.
   - Check and create the `reconstruct` database if it doesn't exist.
   - Save the reconstructed image and prompt to Cloudant in Base64 format.

2. **Inpainting Pipeline**:
   - Load the Stable Diffusion inpainting model using `AutoPipelineForInpainting`.
   - Use PyTorch to ensure compatibility with GPU acceleration.

3. **Gradio Interface**:
   - Use `gr.Blocks` to build a clean and interactive UI.
   - Include components for image upload, prompt input, and result display.

4. **Error Handling**:
   - Catch exceptions for image loading, inpainting, and Cloudant operations.
   - Provide meaningful error messages to the user.

---

## Example Prompt

- **Prompt**: "Concept art digital painting of an elven castle, inspired by Lord of the Rings, highly detailed, 8k"

---

## Troubleshooting

1. **Model Not Found**:
   - Ensure the correct model path and name in `AutoPipelineForInpainting.from_pretrained()`.

2. **Cloudant Errors**:
   - Verify credentials and database URL.

3. **CUDA Not Available**:
   - Install CUDA-compatible PyTorch and drivers, or run on CPU with slower performance.

---

## Contributors

- **Prithviraj Ghorpade** & **Harshal Ghugal**.
