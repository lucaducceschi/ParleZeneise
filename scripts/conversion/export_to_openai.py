import torch
import argparse
from transformers import WhisperForConditionalGeneration

def export_to_pt(hf_model_path, output_path):
    """Convert a HF-style folder (result of trainer.train()) back to a Whisper-loadable .pt file."""
    model = WhisperForConditionalGeneration.from_pretrained(hf_model_path)
    state_dict = model.model.state_dict()
    
    # Optional cleanup: strip 'model.' prefix if coming from a wrapper
    # ...
    
    torch.save({"model_state_dict": state_dict}, output_path)
    print(f"Exported model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--dest", type=str, required=True, help="Output .pt file path")
    args = parser.parse_args()
    export_to_pt(args.src, args.dest)
