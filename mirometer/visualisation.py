import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def add_text_on_mask(image, mask, text):
    """
    Adds a tag to the image at the center of the detected area.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask of the detected area.
        text (str): Text to be added as a tag.

    Returns:
        result_image (numpy.ndarray): Image with the tag added.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask.")

    largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        raise ValueError("Contour area is zero, cannot compute center.")
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    result_image = image.copy()
    cv2.putText(result_image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    return result_image

def visualize_image(image):
    """
    Visualizes the given image.

    Args:
        image (numpy.ndarray): Image to be visualized.
    """
    if image is None:
        raise ValueError("Invalid image provided.")

    cv2.imshow("Image Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_evaluation(area_ratios):
    """
    Plots the evaluation results of area ratios.

    Args:
        area_ratios (list): List of area ratios to be plotted.
    """
    mean = np.mean(area_ratios)
    std = np.std(area_ratios)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title('Area Ratio Distribution on the test set', fontsize=16)

    counts, bins, _ = ax1.hist(area_ratios, bins=30, density=True, alpha=0.6, color='g', label='Density')
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)

    ax1.set_xlabel('Area Ratio')
    ax1.set_ylabel('Probability Density', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True)

    ax1.text(0.95, 0.95, f'Mean: {mean:.4f}\nStd Dev: {std:.4f}',
            horizontalalignment='right', verticalalignment='top',
            transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    ax2 = ax1.twinx()
    ax2.hist(area_ratios, bins=bins, alpha=0.99, color='b', zorder=1, label='Frequency')
    ax2.set_ylabel('Frequency', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    bin_width = bins[1] - bins[0]
    total_count = len(area_ratios)
    ax2.plot(
        x,
        stats.norm.pdf(x, mean, std) * bin_width * total_count, 
        'g-', lw=3, zorder=10, 
        label=f'Normal (μ={mean:.4f}, σ={std:.4f})'
    )
    ax1.axvline(x=1.00, color='r', linestyle='--', lw=1.5, zorder=11, label='Theoretical Area Ratio = 1.00')
    plt.savefig("evaluation_results/area_ratio_distribution.png")
    
    plt.savefig("evaluation_results/area_ratio_distribution.png")