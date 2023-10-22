import matplotlib.pyplot as plt

from PIL import Image

path_to_annotations = "dataset-iiit-pet-master/annotations/trimaps/"
path_to_image = "dataset-iiit-pet-master/images/"

# load image
annotation = Image.open(path_to_annotations + "Abyssinian_1.png")
plt.subplot(1, 2, 1)
plt.title("Annotation")
plt.imshow(annotation)

image = Image.open(path_to_image + "Abyssinian_1.jpg")
plt.subplot(1, 2, 2)
plt.title("Image")
plt.imshow(image)

plt.savefig("plot.png")
plt.show()