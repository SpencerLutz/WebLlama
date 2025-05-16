import torch
import os
import numpy as np

# Load the checkpoint
checkpoint = torch.load('./Llama-3.2-1B/original/consolidated.00.pth',
                       map_location=torch.device('cpu'))

# Create output directory
output_dir = 'weights'
os.makedirs(output_dir, exist_ok=True)

for key, tensor in checkpoint.items():
    # Handle BFloat16 by converting to Float32 first
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)

    # Convert to numpy array
    numpy_array = tensor.numpy()

    # Save as .bin file
    output_path = os.path.join(output_dir, f'{key}.bin')
    numpy_array.tofile(output_path)

    # Save metadata (shape and original dtype)
    with open(os.path.join(output_dir, f'{key}.meta'), 'w') as f:
        f.write(f"shape: {list(tensor.shape)}\n")
        f.write(f"dtype: {str(tensor.dtype).replace('torch.', '')}\n")

    print(f'Saved {key} to {output_path}')

print("Conversion complete. Each tensor saved as .bin with corresponding .meta file.")
