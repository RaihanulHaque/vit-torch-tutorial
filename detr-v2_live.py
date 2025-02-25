import torch
import cv2
import numpy as np
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import time

# Check if MPS is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# Load model and processor
image_processor = RTDetrImageProcessor.from_pretrained("jadechoghari/RT-DETRv2")
model = RTDetrForObjectDetection.from_pretrained("jadechoghari/RT-DETRv2").to(device)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

start_time = time.time()
frame_count = 0

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert BGR (OpenCV format) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for processing
    image = Image.fromarray(frame_rgb)

    # Prepare input for model
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)

    # Draw bounding boxes on the original frame
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]
            
            # Draw rectangle
            cv2.rectangle(frame, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (255, 0, 0),  # Blue in BGR
                         2)
            
            # Add label and score
            label_text = f"{model.config.id2label[label]}: {score:.2f}"
            cv2.putText(frame, label_text, 
                       (int(box[0]), int(box[1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Display the frame
    cv2.imshow("Webcam Object Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()