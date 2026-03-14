"""
Fill masks with encoder model
Use █ or [MASK] to mark positions to fill
"""

import torch

from models.encoder import Encoder

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
model = Encoder().to(device)
model.load_state_dict(torch.load("", map_location=device))
model.eval()

print(f"Loaded model on {device}")
print("Use █ to mark masks (or type [MASK])")
print("Type 'quit' to exit\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "quit":
        break

    # Replace [MASK] with █ for convenience
    prompt = prompt.replace("[MASK]", "█")

    # Convert to tokens, 0 for mask positions
    tokens = [0 if c == "█" else ord(c) for c in prompt]
    x = torch.tensor([tokens]).to(device)

    output = model.fill_masks(x)
    response = "".join(chr(t) for t in output[0].tolist())
    print(f"Model: {response}\n")
