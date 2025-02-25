import torch
import requests
import cv2
import numpy as np
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Check if MPS is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image_path = '/home/rahi/Code/vit_torch/images/dog.jpg'
image = Image.open(image_path)

image_processor = RTDetrImageProcessor.from_pretrained("jadechoghari/RT-DETRv2")
model = RTDetrForObjectDetection.from_pretrained("jadechoghari/RT-DETRv2").to(device)

print(model.config.id2label)

inputs = image_processor(images=image, return_tensors="pt")
# Move inputs to the MPS device
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

# Convert PIL Image to NumPy array for OpenCV
image_np = np.array(image)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
        cv2.rectangle(image_np, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (255, 0, 0),  # BGR color (blue)
                     2)  # thickness
        
        cv2.putText(image_np, 
                    f"{model.config.id2label[label]}: {score:.2f}", 
                    (int(box[0]), int(box[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (36,25,12), 2)

# Show the image
cv2.imshow("image", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()