import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = './14813.jpg' # 이미지 경로

img = cv2.imread(img_path)
img3 = cv2.imread(img_path)

med_val = np.median(img)
lower = int(max(0, 0.7*med_val))
upper = int(min(255, 1.3*med_val))

#dst = cv2.GaussianBlur(img, (3, 3), 0, 0)
dst = cv2.Canny(img, lower, upper, 3)
img2 = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGRA)

_, img_bin = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)


cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)
stats_array=np.array(stats)
max = np.max(stats_array[1:, 4])
print(max)
print(stats_array)

bound_list = []

for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]
    if area < 100 or area >max-10:
        continue
    cv2.rectangle(img, (x, y, w, h), (255, 0, 0))
    cv2.rectangle(img2, (x, y, w, h), (255, 0, 0))
    bound_list.append(stats[i])

bound_array = np.array(bound_list)
bound_array[:, 2] = bound_array[:, 0] +bound_array[:, 2]
bound_array[:, 3] = bound_array[:, 1] +bound_array[:, 3]

print(bound_array)
max_x = np.max(bound_array[:, 2])
min_x = np.min(bound_array[:, 0])
max_y = np.max(bound_array[:, 3])
min_y = np.min(bound_array[:, 1])
width = max_x - min_x
hight = max_y - min_y


region_pixels = img3[min_y:max_y, min_x:max_x]
gray_region_pixels = cv2.cvtColor(region_pixels, cv2.COLOR_BGR2GRAY)
histogram = cv2.calcHist([gray_region_pixels], [0], None, [256], [0, 256])
values_above_threshold = [idx for idx, freq in enumerate(histogram) if freq >= 50]
plt.plot(histogram)
plt.show()
print(values_above_threshold)

region_pixels=cv2.resize(region_pixels, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
region_gray = cv2.cvtColor(region_pixels, cv2.COLOR_RGB2GRAY)
_, img_bin2 = cv2.threshold(region_gray, 50, 255, cv2.THRESH_BINARY)
black_pixel_count = np.sum(img_bin2 == 0)
w,h =img_bin2.shape
print(f'동공비율 : {black_pixel_count}/{w*h}= {black_pixel_count/(w*h)}')
print(f'눈 크기 비 : {hight}/{width} ={hight/width}')
cv2.imshow('region_pixels', region_pixels)
cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.imshow('img_bin', img_bin)
cv2.imshow('img_bin2', img_bin2)
cv2.waitKey()