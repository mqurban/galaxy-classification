import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import os
import random

def create_elliptical_galaxy(size=64):
    """Generates a synthetic elliptical galaxy image."""
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    
    # Random parameters
    center_x = size // 2 + random.randint(-5, 5)
    center_y = size // 2 + random.randint(-5, 5)
    radius_x = random.randint(10, 25)
    radius_y = int(radius_x * random.uniform(0.6, 1.0)) # Ellipticity
    angle = random.randint(0, 180)
    
    # Draw an ellipse
    draw.ellipse(
        [(center_x - radius_x, center_y - radius_y), (center_x + radius_x, center_y + radius_y)],
        fill=random.randint(150, 255),
        outline=None
    )
    
    # Rotate and blur to make it look like a galaxy
    image = image.rotate(angle)
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))
    
    # Add noise
    img_array = np.array(image)
    noise = np.random.normal(0, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return img_array

def create_spiral_galaxy(size=64):
    """Generates a synthetic spiral galaxy image (simplified as a swirly structure)."""
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    
    center_x, center_y = size // 2, size // 2
    
    # Draw spiral arms using points
    num_arms = random.randint(2, 4)
    for i in range(num_arms):
        angle_offset = (2 * np.pi / num_arms) * i
        points = []
        for r in range(5, size // 2 - 5):
            theta = 0.2 * r + angle_offset
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            points.append((x, y))
        
        if len(points) > 2:
            draw.line(points, fill=random.randint(100, 200), width=random.randint(2, 4))

    # Add central bulge
    draw.ellipse(
        [(center_x - 5, center_y - 5), (center_x + 5, center_y + 5)],
        fill=255
    )
    
    # Blur heavily to make it look nebulous
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
    
    # Add noise
    img_array = np.array(image)
    noise = np.random.normal(0, 15, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return img_array

def generate_dataset(num_samples=1000, output_dir="data"):
    """Generates a dataset of synthetic galaxy images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    X = []
    y = [] # 0 for Elliptical, 1 for Spiral
    
    print(f"Generating {num_samples} synthetic galaxy images...")
    
    for i in range(num_samples):
        if i % 2 == 0:
            img = create_elliptical_galaxy()
            label = 0
        else:
            img = create_spiral_galaxy()
            label = 1
        
        X.append(img)
        y.append(label)
        
        # Save sample images for inspection
        if i < 10:
            Image.fromarray(img).save(f"{output_dir}/sample_{i}_label_{label}.png")
            
    X = np.array(X)
    y = np.array(y)
    
    # Normalize pixel values
    X = X / 255.0
    X = X.reshape(-1, 64, 64, 1) # Add channel dimension for CNN
    
    print("Dataset generation complete.")
    return X, y

if __name__ == "__main__":
    generate_dataset(20)
