# A fastAPI code for the model API

import os
import sys
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path


# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

def inference(message:str, image):
    # Preparation for inference
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


# FastAPI
app = FastAPI()

# Create a directory for saving images if it doesn't exist
UPLOAD_DIR = Path("images")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate/")
def generate(message: str = "Describe this image.", image: UploadFile = File(...)):
    # Save the image
    file = UPLOAD_DIR / image.filename

    # Check if the image already exists
    if file.exists():
        abs_path = file.absolute()
    else:
        with open(file, "wb") as f:
            f.write(image.file.read())
        abs_path = file.absolute()
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{abs_path}",
                },
                {"type": "text", "text": message},
            ],
        }
    ]
    return inference(messages, image)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="")