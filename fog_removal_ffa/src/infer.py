import os
import argparse
import yaml
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model.ffa_net import FFA_Net
from utils.metrics import calculate_metrics

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_single_image(model, image_path, output_path, device, image_size=256):
    """Process a single image and save the result"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    inference_time = time.time() - start_time
    
    # Convert to PIL image and resize to original size
    output_tensor = output.squeeze(0).cpu().clamp(0, 1)
    
    # Save as image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(output_tensor, output_path)
    
    print(f"Processed image saved to {output_path}")
    print(f"Inference time: {inference_time*1000:.2f} ms")
    
    return output_path, inference_time

def evaluate_test_set(model, test_dir, output_dir, device, image_size=256):
    """Evaluate the model on the test set"""
    # Check directories
    hazy_dir = os.path.join(test_dir, 'hazy')
    clean_dir = os.path.join(test_dir, 'clean')
    
    if not os.path.exists(hazy_dir) or not os.path.exists(clean_dir):
        raise ValueError(f"Test directory should contain 'hazy' and 'clean' subdirectories")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    hazy_files = sorted([os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir) 
                        if f.endswith(('.jpg', '.png', '.jpeg'))])
    clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Ensure matching pairs
    assert len(hazy_files) == len(clean_files), "Number of hazy and clean images must match"
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Process each image pair
    psnr_values = []
    ssim_values = []
    inference_times = []
    
    for i, (hazy_path, clean_path) in enumerate(tqdm(zip(hazy_files, clean_files), total=len(hazy_files), desc="Processing")):
        # Load images
        hazy_img = Image.open(hazy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Preprocess
        hazy_tensor = transform(hazy_img).unsqueeze(0).to(device)
        clean_tensor = transform(clean_img).unsqueeze(0).to(device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            output = model(hazy_tensor)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Calculate metrics
        metrics = calculate_metrics(output, clean_tensor)
        psnr_values.append(metrics['PSNR'])
        ssim_values.append(metrics['SSIM'])
        
        # Save output
        output_path = os.path.join(output_dir, f"dehazed_{os.path.basename(hazy_path)}")
        save_image(output.squeeze(0).cpu().clamp(0, 1), output_path)
        
        # Save comparison (hazy | dehazed | clean)
        comparison = torch.cat([hazy_tensor.cpu(), output.cpu(), clean_tensor.cpu()], dim=3)
        comparison_path = os.path.join(output_dir, f"comparison_{os.path.basename(hazy_path)}")
        save_image(comparison.squeeze(0).clamp(0, 1), comparison_path)
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_time = np.mean(inference_times)
    
    # Print and save metrics
    print(f"\nTest Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average inference time: {avg_time*1000:.2f} ms\n")
        
        # Per-image results
        f.write("\nPer-image results:\n")
        for i, (hazy_file, psnr, ssim) in enumerate(zip(hazy_files, psnr_values, ssim_values)):
            f.write(f"{os.path.basename(hazy_file)}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}\n")
    
    print(f"Metrics saved to {metrics_file}")
    return avg_psnr, avg_ssim

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference with FFA-Net for fog removal")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single input image')
    parser.add_argument('--output', type=str, help='Path to save output for single image')
    parser.add_argument('--test_dir', type=str, help='Path to test directory with hazy and clean subdirectories')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = FFA_Net(
        in_channels=config['model']['in_channels'],
        num_features=config['model']['num_features'],
        num_groups=config['model']['num_groups'],
        num_blocks=config['model']['num_blocks']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Process single image or test directory
    if args.image:
        # Process single image
        if args.output is None:
            args.output = os.path.join(config['paths']['results_dir'], 'dehazed_output.png')
        
        process_single_image(
            model=model,
            image_path=args.image,
            output_path=args.output,
            device=device,
            image_size=config['dataset']['image_size']
        )
    elif args.test_dir:
        # Process test directory
        if args.output_dir is None:
            args.output_dir = os.path.join(config['paths']['results_dir'], 'test_results')
        
        evaluate_test_set(
            model=model,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            device=device,
            image_size=config['dataset']['image_size']
        )
    else:
        print("Please provide either --image or --test_dir argument")

if __name__ == "__main__":
    main()