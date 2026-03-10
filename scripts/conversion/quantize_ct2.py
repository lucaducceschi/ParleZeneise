import os
import subprocess
import argparse

def quantize_model(hf_model_path, output_dir, quantization="int8"):
    """
    Quantize a HuggingFace Whisper model to CTranslate2 format.
    Requires: ctranslate2 >= 3.0.0
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cmd = [
        "ct2-transformers-converter",
        "--model", hf_model_path,
        "--output_dir", output_dir,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", quantization,
        "--force"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully quantized model to {output_dir}")
    else:
        print(f"Quantization failed with error: {result.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="HF model directory")
    parser.add_argument("--dest", type=str, required=True, help="CTranslate2 output folder")
    parser.add_argument("--quant", type=str, choices=["int8", "float16", "int16"], default="int8")
    args = parser.parse_args()
    quantize_model(args.src, args.dest, args.quant)
