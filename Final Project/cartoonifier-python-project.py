import cv2 
import easygui 
import numpy as np 
import imageio 

import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import messagebox, Label, Button
from PIL import ImageTk, Image


global ReSized6 
global process_images 
global final_image_label 


top=tk.Tk()
top.geometry('800x600') 
top.title('Cartoonify Your Image !')
top.configure(background='white')


final_image_label = Label(top, background='white')
final_image_label.pack(pady=10)



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
                  "4. Edges (Threshold)", "5. Color Quantized", "6. Final Cartoon"] 
                  
        for i, ax in enumerate(axes.flat):
            
            ax.imshow(process_images[i], cmap='gray' if i in [1, 2, 3] else None) 
            ax.set_title(titles[i], fontsize=8)

        plt.show()
    else:
        messagebox.showinfo(title="Error", message="Please cartoonify an image first.")



def cartoonify(ImagePath):
    """Processes the image and displays the final cartoon result with an enhanced cartoon style."""
    global ReSized6
    global process_images
    

    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)

    if originalmage is None:
        messagebox.showerror("Error", "Can not find any image. Choose appropriate file")
        return

 
    ReSized1 = cv2.resize(originalmage, (500, 300)) 

    
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (500, 300))

    
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, (500, 300))

    
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 11, 12) 
    ReSized4 = cv2.resize(getEdge, (500, 300))
    
    
    data = np.float32(originalmage).reshape(-1, 3)

    
    k = 8 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
   
    center = np.uint8(center)
    quantized_image = center[label.flatten()]
    quantized_image = quantized_image.reshape(originalmage.shape)
    
    
    ReSized5 = cv2.resize(quantized_image, (500, 300))

   
    inverted_edge = cv2.bitwise_not(getEdge)
    

    edges_colored = np.zeros_like(quantized_image) 
    
    
    cartoonImage = cv2.bitwise_and(quantized_image, quantized_image, mask=getEdge)

    
    ReSized6 = cv2.resize(cartoonImage, (500, 300))

   
    process_images = [ReSized1, ReSized2, ReSized3, ReSized4, ReSized5, ReSized6]


    img_pil = Image.fromarray(ReSized6)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    final_image_label.config(image=img_tk)
    final_image_label.image = img_tk 
    
   
    save_button.pack_forget() 
    process_button.pack_forget() 
    
    save_button.pack(side=tk.TOP, pady=10)
    process_button.pack(side=tk.TOP, pady=10)


def save():
    """Saves the final cartoon image to the user's computer."""
    global ReSized6
    if 'ReSized6' in globals():
     
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"), 
                                                       ("JPEG files", "*.jpg"), 
                                                       ("All files", "*.*")])
        if path:
            
            cv2.imwrite(path, cv2.cvtColor(ReSized6, cv2.COLOR_RGB2BGR))
            newName = os.path.basename(path)
            I = f"Image saved successfully as {newName}"
            tk.messagebox.showinfo(title="Image Saved", message=I)
    else:
        messagebox.showinfo(title="Error", message="No cartoon image to save yet.")




upload_button = Button(top, text="🖼️ Cartoonify an Image", command=upload, padx=10, pady=5)
upload_button.configure(background='#364156', foreground='white', font=('calibri', 12, 'bold'))
upload_button.pack(side=tk.TOP, pady=20) 


save_button = Button(top, text="💾 Save Cartoon Image", command=save, padx=10, pady=5)
save_button.configure(background='#364156', foreground='white', font=('calibri', 12, 'bold'))

process_button = Button(top, text="📈 View Process Steps", command=view_process, padx=10, pady=5)
process_button.configure(background='#364156', foreground='white', font=('calibri', 12, 'bold'))


top.mainloop()