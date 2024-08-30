import cv2
from numpy import random
import numpy as np

#-----------------------------------------------------------------
# MSE - Mean Square Error
# Process pixel by pixel
def MSE(image_a, image_b):
    w, h, c = image_a.shape
    MSE = 0.0
    for w_i in range(w):
        for h_i in range(h):
            # Current pixel's squared difference
            dif2_r = (int(image_a[w_i, h_i, 0]) - int(image_b[w_i, h_i, 0])) ** 2
            dif2_g = (int(image_a[w_i, h_i, 1]) - int(image_b[w_i, h_i, 1])) ** 2
            dif2_b = (int(image_a[w_i, h_i, 2]) - int(image_b[w_i, h_i, 2])) ** 2
            
            sum_of_diff = dif2_r + dif2_g + dif2_b
            
            # Calculate sum of all pixels
            MSE += sum_of_diff
            
    MSE /= (w * h)
    return MSE

#-----------------------------------------------------------------
def ColourPaletteGeneration(image_in, epochs=10):
    w, h, c = image_in.shape
    
    # Create codebook (colour palette)
    codebook = random.rand(8, 8, 3) * 255
    
    learn_rate = 1.0
    learn_rate_step = 1.0 / (epochs * 256)
    
    for epoch in range(epochs):
        for l in range(h * w):
            randomPixelRow = int(random.rand() * h)
            randomPixelCol = int(random.rand() * w)
            currentPixel = image_in[randomPixelRow, randomPixelCol, :]

            # Find winner codeword
            minDist = float('inf')
            minIndex_r = 0
            minIndex_c = 0
            for r in range(8):
                for c in range(8):
                    dist = np.sqrt(
                        (currentPixel[0] - codebook[r, c, 0]) ** 2 + 
                        (currentPixel[1] - codebook[r, c, 1]) ** 2 + 
                        (currentPixel[2] - codebook[r, c, 2]) ** 2
                    )
                    if dist < minDist:
                        minDist = dist
                        minIndex_r = r
                        minIndex_c = c

            # Update winner's weight
            codebook[minIndex_r, minIndex_c, :] += learn_rate * (currentPixel - codebook[minIndex_r, minIndex_c, :])

        # ลด learning rate
        learn_rate -= learn_rate_step
        print(f"Epoch {epoch + 1}, Learning Rate: {learn_rate:.4f}")
        print("Codebook after update:", codebook)

    return codebook

#-----------------------------------------------------------------
def PixelMapping(image_in, codebook):
    w, h, c = image_in.shape
    image_out = np.zeros((w, h, 3), dtype=np.uint8)
    
    # At this point, we have a codebook whose all members are 
    # between [0, 255]
    # Visit pixel in the image one by one **** Randomly ****
    for r_c in range(h):
        for c_c in range(w):
            currentPixel = image_in[r_c, c_c, :]

            # Find winner codeword
            minDist = 100000
            minIndex_r = 0
            minIndex_c = 0
            for r in range(8):
                for c in range(8):
                    dist = np.sqrt(
                        (currentPixel[0] - codebook[r, c, 0]) ** 2 + 
                        (currentPixel[1] - codebook[r, c, 1]) ** 2 + 
                        (currentPixel[2] - codebook[r, c, 2]) ** 2
                    )
                    if dist < minDist:
                        minDist = dist
                        minIndex_r = r
                        minIndex_c = c

            image_out[r_c, c_c, :] = codebook[minIndex_r, minIndex_c, :]
    
    return image_out

# อ่านภาพที่ให้มา
image = cv2.imread('images\monkey.jpeg')

# ตรวจสอบว่าไฟล์ภาพถูกอ่านได้หรือไม่
if image is None:
    raise ValueError("Image not found. Please check the file path.")

# เปลี่ยนภาพจาก BGR เป็น RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ตรวจสอบขนาดของภาพ
print("Original Image shape:", image.shape)

# สร้าง Colour Palette
codebook = ColourPaletteGeneration(image)

# ตรวจสอบว่า codebook ถูกสร้างขึ้นมาอย่างถูกต้อง
if codebook is None:
    raise ValueError("Codebook is None. There might be an issue with the ColourPaletteGeneration function.")

# ทำ Image Quantization
quantized_image = PixelMapping(image, codebook)

# ตรวจสอบขนาดของภาพที่ผ่านการ Quantization
print("Quantized Image shape:", quantized_image.shape)

# คำนวณค่า MSE
mse_value = MSE(image, quantized_image)
print("MSE:", mse_value)

# แสดงภาพต้นฉบับและภาพที่ผ่านการ Quantization
cv2.imshow("Original Image", image)
cv2.imshow("Quantized Image", quantized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
