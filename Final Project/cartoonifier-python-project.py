import cv2 # for image processing
import easygui # to open the filebox
import numpy as np # to store image
import imageio # to read image stored at particular path

import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import messagebox, Label, Button
from PIL import ImageTk, Image

# Global variables to store image data and widget references
global ReSized6 # To store the final cartoon image for saving
global process_images # To store the list of 6 images for the process view
global final_image_label # To reference the label displaying the final image

# Initialize main window
top=tk.Tk()
top.geometry('800x600') # Increased size for better display
top.title('Cartoonify Your Image !')
top.configure(background='white')

# Label for displaying the final cartoon image (placeholder)
final_image_label = Label(top, background='white')
final_image_label.pack(pady=10)

# --- Button Functions ---

def upload():
    """Opens a file dialog for the user to select an image."""
    ImagePath=easygui.fileopenbox()
    if ImagePath:
        cartoonify(ImagePath)

def view_process():
    """Displays the Matplotlib plot of the 6-step cartoonification process."""
    if 'process_images' in globals():
        fig, axes = plt.subplots(3, 2, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
        fig.suptitle('Cartoonification Process Steps', fontsize=16)
        
        titles = ["1. Original", "2. Grayscale", "3. Smoothed Grayscale", 
                  "4. Edges (Threshold)", "5. Color Quantized", "6. Final Cartoon"] # Updated title for step 5
                  
        for i, ax in enumerate(axes.flat):
            # Use cmap='gray' for monochrome images (Steps 2, 3, 4)
            ax.imshow(process_images[i], cmap='gray' if i in [1, 2, 3] else None) 
            ax.set_title(titles[i], fontsize=8)

        plt.show()
    else:
        messagebox.showinfo(title="Error", message="Please cartoonify an image first.")

# --- Main Image Processing Function ---

def cartoonify(ImagePath):
    """Processes the image and displays the final cartoon result with an enhanced cartoon style."""
    global ReSized6
    global process_images
    
    # Read and convert the image
    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)

    if originalmage is None:
        messagebox.showerror("Error", "Can not find any image. Choose appropriate file")
        return

    # 1. Processing Steps (Resized for consistent display)
    ReSized1 = cv2.resize(originalmage, (500, 300)) # Original

    # 2. Grayscale
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (500, 300))

    # 3. Smoothening (for edge detection)
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, (500, 300))

    # 4. Edge retrieval (Stronger Thresholding for bolder lines)
    # Increased block size and C value for bolder lines
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 11, 12) # Adjusted block size (11) and C (12)
    ReSized4 = cv2.resize(getEdge, (500, 300))
    
    # --- NEW: Color Quantization for flatter colors (Anime/Cartoon look) ---
    # Convert original image to float32 for K-Means
    data = np.float32(originalmage).reshape(-1, 3)

    # Define criteria and apply K-Means clustering
    # Reduce colors to a smaller palette (e.g., 8-16 colors)
    k = 8 # Number of colors you want to quantize to
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    center = np.uint8(center)
    quantized_image = center[label.flatten()]
    quantized_image = quantized_image.reshape(originalmage.shape)
    
    # 5. Store the Color Quantized image for process view
    ReSized5 = cv2.resize(quantized_image, (500, 300))

    # 6. Mask edges with our Color Quantized image (Final Cartoon)
    # The `getEdge` mask is black (0) for edges and white (255) for non-edges.
    # We want edges to be black in the final image, so we use `bitwise_and`
    # and then paint the edges over the quantized image.
    
    # Invert the edge mask: white for edges, black for non-edges
    inverted_edge = cv2.bitwise_not(getEdge)
    
    # Create a blank image to draw black lines on
    edges_colored = np.zeros_like(quantized_image) 
    
    # Use the inverted_edge as a mask to draw black lines on the edges_colored image
    # For every pixel where inverted_edge is white (an edge), keep it black in edges_colored
    # For every pixel where inverted_edge is black (not an edge), keep it black in edges_colored
    # This just gives us a black background with black lines on edges, effectively
    # we need the edges from getEdge to be painted over the quantized image.

    # A simpler way: use getEdge directly. Where getEdge is black (0), the output will be black.
    # Where getEdge is white (255), the output will be the quantized_image pixel.
    # This effectively "draws" black lines (where getEdge is 0) over the quantized image.
    cartoonImage = cv2.bitwise_and(quantized_image, quantized_image, mask=getEdge)

    # If getEdge is black (0) at an edge, the result of bitwise_and is black.
    # If getEdge is white (255) elsewhere, the result is the quantized color.
    # This creates the desired effect of black outlines on flat colors.
    
    ReSized6 = cv2.resize(cartoonImage, (500, 300))

    # Store all images for the process view function
    process_images = [ReSized1, ReSized2, ReSized3, ReSized4, ReSized5, ReSized6]

    # Display the final cartoon image in the main GUI window
    img_pil = Image.fromarray(ReSized6)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    final_image_label.config(image=img_tk)
    final_image_label.image = img_tk # Keep a reference to prevent garbage collection!
    
    # Ensure "Save" and "View Process" buttons are packed after processing
    # We use pack_forget and then pack to ensure they are at the bottom regardless of previous state
    save_button.pack_forget() 
    process_button.pack_forget() 
    
    save_button.pack(side=tk.TOP, pady=10)
    process_button.pack(side=tk.TOP, pady=10)


def save():
    """Saves the final cartoon image to the user's computer."""
    global ReSized6
    if 'ReSized6' in globals():
        # Open a Save As dialog
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"), 
                                                       ("JPEG files", "*.jpg"), 
                                                       ("All files", "*.*")])
        if path:
            # Convert RGB back to BGR before saving with cv2
            cv2.imwrite(path, cv2.cvtColor(ReSized6, cv2.COLOR_RGB2BGR))
            newName = os.path.basename(path)
            I = f"Image saved successfully as {newName}"
            tk.messagebox.showinfo(title="Image Saved", message=I)
    else:
        messagebox.showinfo(title="Error", message="No cartoon image to save yet.")


# --- GUI Buttons Setup ---

upload_button = Button(top, text="🖼️ Cartoonify an Image", command=upload, padx=10, pady=5)
upload_button.configure(background='#364156', foreground='white', font=('calibri', 12, 'bold'))
upload_button.pack(side=tk.TOP, pady=20) 

# Save button (created, will be packed after image processing)
save_button = Button(top, text="💾 Save Cartoon Image", command=save, padx=10, pady=5)
save_button.configure(background='#364156', foreground='white', font=('calibri', 12, 'bold'))

# View Process button (created, will be packed after image processing)
process_button = Button(top, text="📈 View Process Steps", command=view_process, padx=10, pady=5)
process_button.configure(background='#364156', foreground='white', font=('calibri', 12, 'bold'))


top.mainloop()