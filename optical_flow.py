import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_base_image(shape, num_shapes=10, seed=42):
    np.random.seed(seed)
    img = np.zeros(shape, dtype=np.uint8)
    shapes = []
    for _ in range(num_shapes):
        center = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
        radius = np.random.randint(5, 30)
        color = np.random.randint(100, 256)
        cv2.circle(img, center, radius, (color, color, color), -1)
        shapes.append((center, radius, color))
    return img, shapes

def generate_moved_image(shape, shapes, movement_range=10, seed=42):
    np.random.seed(seed)
    img = np.zeros(shape, dtype=np.uint8)
    for center, radius, color in shapes:
        new_center = (center[0] + np.random.randint(-movement_range, movement_range),
                      center[1] + np.random.randint(-movement_range, movement_range))
        cv2.circle(img, new_center, radius, (color, color, color), -1)
    return img

def compute_optical_flow(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Compute magnitude and angle of 2D vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image to visualize flow
    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow_img, flow, mag, hsv

def overlay_images(img1, img2, alpha=0.5):
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return vis

# Generate base image and shapes
base_img, shapes = generate_base_image((300, 300, 3))

# Generate moved image with some movements
moved_img = generate_moved_image((300, 300, 3), shapes)

# Compute optical flow and magnitudes
flow_img, flow, mag, hsv = compute_optical_flow(base_img, moved_img)

# Overlay images
overlay_img = overlay_images(base_img, moved_img)

# Draw flow on the overlay image
flow_overlay_img = draw_flow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY), flow)

# Plot the original images and their overlay
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Image 1")
plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Image 2")
plt.imshow(cv2.cvtColor(moved_img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

# Plot the optical flow visualizations
plt.figure(figsize=(24, 6))

plt.subplot(1, 5, 1)
plt.title("Flow with Directions")
plt.imshow(cv2.cvtColor(flow_overlay_img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 5, 2)
plt.title("Optical Flow - H Channel")
plt.imshow(hsv[..., 0], cmap='hsv')
plt.colorbar()

plt.subplot(1, 5, 3)
plt.title("Optical Flow - S Channel")
plt.imshow(hsv[..., 1], cmap='gray')
plt.colorbar()

plt.subplot(1, 5, 4)
plt.title("Optical Flow - V Channel")
plt.imshow(hsv[..., 2], cmap='gray')
plt.colorbar()

plt.subplot(1, 5, 5)
plt.title("Flow Magnitude")
plt.imshow(mag, cmap='hot')
plt.colorbar()

plt.tight_layout()
plt.show()

# Analyze magnitudes
print("Max flow magnitude:", np.max(mag))
print("Min flow magnitude:", np.min(mag))
