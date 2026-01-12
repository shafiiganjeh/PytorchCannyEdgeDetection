import cv2
import torch
import numpy as np
import time
from torchvision import transforms
import edge_detector as ce
from torchvision.transforms import v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Press 'q' to quit")

getedge = ce.c_edge(upper_treshold = 20,lower_treshold = 10)
getedge.to(device).compile()

height = 600
width = 800

preprocess = transforms.Compose([
    
    transforms.ToPILImage(),
    transforms.Resize((width, height)), 
    transforms.ToTensor(),
    v2.Grayscale()
])

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps_time = time.time()
    frame_count = 0
    
    with torch.no_grad():  # Disable gradient calculation for inference
        while True:
            ret, frame = cap.read()
            
            # frame capture
            if not ret:
                print("Error: Failed to capture frame")
                break
            original_frame = frame.copy()
            input_tensor = preprocess(frame)
            input_batch = input_tensor.unsqueeze(0).to(device)  
            
            # inference
            start_time = time.time()
            input_batch= v2.Grayscale()(input_batch)
            input_batch = input_batch*255
            output = getedge(input_batch)
            inference_time = time.time() - start_time 
            
            output_np = output.squeeze().cpu().numpy()
            
            if output_np.max() > 1.0:
                # Already in 0-255 range
                output_np = output_np.astype(np.uint8)
            else:
                # Scale from 0-1 to 0-255
                output_np = (output_np * 255).astype(np.uint8)
            
            output_resized = cv2.resize(output_np, (original_frame.shape[1], original_frame.shape[0]))
            output_resized = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)
            
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count / (time.time() - fps_time)
                frame_count = 0
                fps_time = time.time()
                print(f"FPS: {fps:.2f}, Inference time: {inference_time*1000:.2f}ms")
                
            cv2.putText(original_frame, f"Input", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_resized, f"Edges", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            top = np.hstack((original_frame, output_resized))
            cv2.imshow('Webcam - Input | Edges | Overlay', top)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released")
 