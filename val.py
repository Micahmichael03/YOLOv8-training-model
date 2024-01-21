from ultralytics import YOLO

# Step 1: Load the trained model
model = YOLO("./runs/detect/train/weights/best.pt")  # Load the trained model
# Replace "yolov8n.pt" with the correct path to your trained model weights.

# Specify the configuration file for validation
val_config = "config.yaml"

# Step 2: Evaluate the model on a validation dataset
metrics = model.val(val_config)  # Evaluate model performance on the validation set

# Step 3: Make predictions on an image
image_path = "./videos1/pure5.jpg"  # Replace with the path to your image
results = model(image_path)  # Predict on an image

# Step 4: Export the model to ONNX format (optional)
onnx_path = model.export(format="onnx")  # Export the model to ONNX format
# The 'onnx_path' variable will contain the path to the exported ONNX model.

# Print the evaluation metrics
print("Evaluation Metrics:")
print(metrics)

# Display the detection results
print("Detection Results:")
results.show()

# You can access the predictions in 'results' for further processing or visualization.
