# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** DHARSHINI K 
- **Register Number:** 212223230047

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
img = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_GRAYSCALE)
```

#### 2. Print the image width, height & Channel.
```python
image = cv2.imread('Eagle_in_Flight.jpg')
print("Height, Width and Channel:", image.shape)
```

![image](https://github.com/user-attachments/assets/e37baac5-5bbc-4e6b-b5e9-190e17ebab98)

#### 3. Display the image using matplotlib imshow().
```python
import matplotlib.pyplot as plt
plt.imshow(img)
```

![image](https://github.com/user-attachments/assets/4ef853ef-38e8-444f-b3a5-6d5d96404534)

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight.png', image)
```

![image](https://github.com/user-attachments/assets/43cfefb1-72d0-4dd2-a3c8-60b2da8fff1c)

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img = cv2.imread('Eagle_in_Flight.png')
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(color_img)
```

![image](https://github.com/user-attachments/assets/d787e668-c54e-4dff-beae-c4c6b1e5b167)

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(color_img)
color_img.shape
```

![image](https://github.com/user-attachments/assets/4cd8721e-ae86-44f7-9f81-dbc63af9f1aa)

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
cropped = color_img[10:450, 150:570]
plt.imshow(cropped)
plt.axis("off")
```

![image](https://github.com/user-attachments/assets/58b7958b-e743-431e-8289-90a7adcd21bf)

#### 8. Resize the image up by a factor of 2x.
```python
height, width = image.shape[:2]
resized_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
plt.imshow(resized_image)
```

![image](https://github.com/user-attachments/assets/bca573cd-d180-4b75-9239-ee94e2f5296f)

#### 9. Flip the cropped/resized image horizontally.
```python
flipped = cv2.flip(resized_image, 1)
plt.imshow(flipped)
```

![image](https://github.com/user-attachments/assets/eda3fc91-111a-48a1-9b5a-bea95921a161)

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img_apollo = cv2.imread('Apollo-11-launch.jpg')
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
cv2.putText(img_apollo, 'Apollo 11 Saturn V Launch, July 16, 1969', (50, img_apollo.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
plt.imshow(img_apollo)
```

![image](https://github.com/user-attachments/assets/33c62560-e869-4068-a527-cd4e3fc68d05)

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
cv2.rectangle(img_apollo, (400, 30), (750, 600), (255, 0, 255), 3)
```

![image](https://github.com/user-attachments/assets/ba72ee94-a6a6-4d76-8bf9-c908062ae26b)

#### 13. Display the final annotated image.
```python
plt.imshow(img_apollo)
```

![image](https://github.com/user-attachments/assets/73be99c9-99d6-488f-b222-44b38510634b)

#### 14. Read the image ('Boy.jpg').
```python
boy_img = cv2.imread('Boy.jpg')
```

#### 15. Adjust the brightness of the image.
```python
# Create a matrix of ones (with data type float64)
import numpy as np
matrix_ones = np.ones(boy_img.shape, dtype='uint8') * 50
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_darker, img_brighter]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

![image](https://github.com/user-attachments/assets/e7b8d28f-1246-4bf0-a8cd-29cec00850af)

#### 18. Modify the image contrast.
```python
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
matrix1 = np.ones(boy_img.shape, dtype='uint8') * 25
matrix2 = np.ones(boy_img.shape, dtype='uint8') * 50
img_higher1 = cv2.addWeighted(boy_img, 1.1, matrix1, 0, 0)
img_higher2 = cv2.addWeighted(boy_img, 1.2, matrix2, 0, 0)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_higher1, img_higher2]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

![image](https://github.com/user-attachments/assets/4b9e3953-8f80-4bb4-b1c4-a81e4b66a55d)

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(boy_img)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([b, g, r]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```

![image](https://github.com/user-attachments/assets/7f989e41-1971-4acc-bd7c-738a3f7af507)

#### 21. Merged the R, G, B , displays along with the original image
```python
merged = cv2.merge([b, g, r])
plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
plt.show()
```

![image](https://github.com/user-attachments/assets/eaedf4fb-7d95-42e3-b59b-56eb3a98502b)

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv = cv2.cvtColor(boy_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([h, s, v]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```

![image](https://github.com/user-attachments/assets/77bf339d-6d39-4ae7-9236-1e05a68753d1)

#### 23. Merged the H, S, V, displays along with original image.
```python
#merged_hsv = cv2.merge([h, s, v])
plt.imshow(cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB))
plt.show()
```

![image](https://github.com/user-attachments/assets/e9b49f24-3f8f-4b9c-be6c-a342b91d6710)

## Output:
- **i)** Read and Display an Image.

![image](https://github.com/user-attachments/assets/2760e694-95a9-434f-8d95-91ffde91b6f2)

- **ii)** Adjust Image Brightness.

![image](https://github.com/user-attachments/assets/95376c59-c9eb-4b21-b23f-110e36b1762a)

- **iii)** Modify Image Contrast.

![image](https://github.com/user-attachments/assets/68f9fa12-afb2-43ed-8908-aecb5c1c0913)
  
- **iv)** Generate Third Image Using Bitwise Operations.

![image](https://github.com/user-attachments/assets/2d9fac34-adbf-4e3e-a0ca-8c75f3d0b133)

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.
