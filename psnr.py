from math import log10, sqrt
import cv2
import numpy as np
import skimage.metrics
  
def PSNR(hazy, dehazy):
	print(hazy,dehazy)
	mse = np.mean((hazy - dehazy) ** 2)
	print("MSE :" + str(mse))
	if(mse == 0):
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def ssim(hazy,dehazy):
	ssim = skimage.metrics.structural_similarity(hazy, dehazy, multichannel=True)
	return ssim

def main():
     hazy = cv2.imread("./haze2.jpg")
     dehazy = cv2.imread("./clear2.jpg", 1)
     hazy = cv2.resize(hazy, (1000, 1000))
     dehazy = cv2.resize(dehazy, (1000, 1000))
     value = PSNR(hazy, dehazy)

     print(f"PSNR value is {value} dB")
     print("ssim is :"  + str(ssim(hazy,dehazy)))
       
if __name__ == "__main__":
    main()