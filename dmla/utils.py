import matplotlib.pyplot as plt

def plot_images(original_image, preproc_image):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].axis("off")  # Turn off the axis
    axes[0].set_title("Original image")  # Set a title

    # Show the second image on the second subplot
    axes[1].imshow(preproc_image)
    axes[1].axis("off")  # Turn off the axis
    axes[1].set_title("Preprocessed image")  # Set a title

    # Display the images
    plt.tight_layout()  # Adjust spacing between subplots

    return plt.show()
