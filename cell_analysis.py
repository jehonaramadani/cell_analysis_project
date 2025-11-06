import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, measure
from skimage.io import imread


#bild hochladen
image =  imread('https://www.sysmex.de/fileadmin/media/f100/images/CellImages/2009_03_figure3_hires.jpg')
rgb_image = image
gray = color.rgb2gray(rgb_image)


plt.figure(figsize=(6,6))
plt.title("Originalbild")
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

#thresholder
thresh = filters.threshold_otsu(gray)
binary = gray > thresh
binary = morphology.remove_small_objects(binary, 100)
binary = morphology.remove_small_holes(binary, 100)

plt.figure(figsize=(6,6))
plt.title("Segmentierte Zellen (Binary Image)")
plt.imshow(binary, cmap='gray')
plt.axis('off')
plt.show()

#extraktion
labels = measure.label(binary)
props = measure.regionprops(labels)

cell_count = len(props)
mean_area = np.mean([p.area for p in props])

print(f"Zellanzahl: {cell_count}")
print(f"Durchschnittliche Zellgröße: {mean_area:.2f} Pixel²")

#vizualisieren
plt.figure(figsize=(6,6))
plt.title("Graustufenbild")
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

for p in props:
    y, x = p.centroid
    ax.plot(x, y, 'r.', markersize=5)  # type: ignore

plt.axis('off')
plt.show()