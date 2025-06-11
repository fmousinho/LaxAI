import os
import torch
import cv2
import numpy as np
from rfdetr import RFDETRBase   # Adjust import based on your actual model structure


# Create a dummy model for demonstration - define it globally
# If you have your actual model, define or import it here.
# from your_model_definition import RF_DETRBase

#model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU())

# Define target image dimensions (adjust if your model expects different dimensions)
TARGET_HEIGHT = 720
TARGET_WIDTH = 1080

def load_and_preprocess_image_to_chw_tensor(image_path: str, height: int, width: int) -> torch.Tensor:
    """
    Loads an image, resizes it, converts to RGB, normalizes, and returns as a CHW PyTorch tensor.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize if necessary
    if img_rgb.shape[0] != height or img_rgb.shape[1] != width:
        img_resized_rgb = cv2.resize(img_rgb, (width, height))
    else:
        img_resized_rgb = img_rgb

    # Convert to PyTorch tensor, normalize to [0, 1], and change to CHW format
    # From (H, W, C) to (C, H, W)
    img_tensor_chw = torch.from_numpy(img_resized_rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
    return img_tensor_chw


# 1. Set the device to MPS
if not torch.backends.mps.is_available():
    print("MPS not available on this device. Cannot perform MPS batch size estimation.")
else:
    device = torch.device("mps")
    print(f"Using device: {device}")

    # 2. Initialize model and move to MPS device
    model = RFDETRBase(device=device.type)  # Replace with your actual model initializatio

    # Define the path to your dummy image
    dummy_image_path = '/Users/fernandomousinho/Library/CloudStorage/GoogleDrive-fmousinho76@gmail.com/My Drive/Colab_Notebooks/Girls-Lacrosse-8/test/Urban_Elite_002_mp4-0000_jpg.rf.fa05304c843e7bb57e8479f8fd83f7c1.jpg'

    # Load and preprocess the base dummy image once
    try:
        base_image_tensor_chw = load_and_preprocess_image_to_chw_tensor(dummy_image_path, TARGET_HEIGHT, TARGET_WIDTH)
        print(f"Successfully loaded and preprocessed base image from: {dummy_image_path}")
    except Exception as e:
        print(f"Error loading or preprocessing base image: {e}")
        exit(1) # Exit if the base image cannot be prepared

    def test_batch_size_mps(batch_size):
        print(f"--- Testing Batch Size: {batch_size} ---")
        torch_input = None # This will hold the batched tensor
        try:
            # 3. Create the batch tensor by repeating the base_image_tensor_chw
            # base_image_tensor_chw is (C,H,W). Unsqueeze to (1,C,H,W) then repeat.
            torch_input = base_image_tensor_chw.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

            # 4. Use torch.no_grad() for inference
            with torch.no_grad():
                _ = model.predict(torch_input) # Pass the batched tensor

            print(f"✅ Batch size {batch_size} fits in memory.")
            return True

        except RuntimeError as e:
            # 5. Catch the MPS-specific error
            if "out of memory" in str(e).lower() or "failed to allocate" in str(e).lower():
                print(f"❌ MPS out of memory with batch size {batch_size}.")
                return False
            else:
                # Re-raise other runtime errors
                print(f"An unexpected RuntimeError occurred with batch size {batch_size}: {e}")
                raise e
        except Exception as e:
            print(f"An unexpected error occurred with batch size {batch_size}: {e}")
            raise e # Re-raise other exceptions
        finally:
            if torch_input is not None:
                del torch_input # Clean up the batch tensor from memory
            # 6. Use the MPS-specific cache empty
            torch.mps.empty_cache()

    # --- Testing Loop ---
    # Start with a batch size of 1 and increase it
    # You can use powers of 2 (1, 2, 4, 8...) or a more linear increment
    print("\nStarting MPS batch size estimation loop...")
    batch_size = 1
    max_successful_batch_size = 0
    try:
        while True:
            if test_batch_size_mps(batch_size):
                max_successful_batch_size = batch_size
                batch_size += 1 # Or batch_size *= 2 for faster searching
                if batch_size > 256: # Safety break for very large batch sizes
                    print(f"Reached safety limit for batch size ({batch_size-1}). Stopping.")
                    break
            else:
                break # Stop if test_batch_size_mps returns False (OOM)
    except Exception as e:
        print(f"Error during batch size testing loop: {e}")
    finally:
        print(f"\n--- MPS Batch Size Estimation Summary ---")
        print(f"Maximum successful batch size found on MPS: {max_successful_batch_size}")
        print("MPS batch size estimation finished.")