"""
Chat with decoder model
"""

import torch

from models.decoder import Decoder

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
model = Decoder().to(device)
model.load_state_dict(torch.load("", map_location=device))
model.eval()

print(f"Loaded model on {device}")
print("Type 'quit' to exit\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "quit":
        break

    tokens = torch.tensor([[ord(c) for c in prompt]]).to(device)
    output = model.generate(tokens, max_tokens=200, top_k=1)
    response = "".join(chr(t) for t in output[0].tolist())
    print(f"Model: {response}\n")
