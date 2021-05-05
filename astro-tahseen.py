# -*- coding: utf-8 -*-
"""
Created on Wed May 5 11:41:51 2021
@author: Tahseen Hamid
"""

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats
import cv2

hdulist = fits.open("mosaic.fits")
header = hdulist[0].header
raw = hdulist[0].data
hdulist.close()

raw_list = raw.ravel()
raw_list_bg = [x for x in raw_list if 3300 <= x <= 3800]

fsize_scale=1
fsize=(6.4*fsize_scale,4.8*fsize_scale)

plt.figure(figsize=(fsize))
plt.title('Full Image Histogram')
plt.xlabel('Pixel Value l')
plt.ylabel('Number of Pixels n')
plt.hist(raw_list, bins=250)
plt.savefig("histo_full.png", dpi=600)
plt.show()

plt.figure(figsize=(fsize))
plt.title('Image Background Histogram (Normalised)')
plt.xlabel('Pixel Value l')
plt.ylabel('Number of Pixels n (Normalised)')
_, bins, _ = plt.hist(raw_list_bg, bins=250, density=True)
mu, sigma = scipy.stats.norm.fit(raw_list_bg)
gaussian = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, gaussian, color='red', linewidth=1, alpha = 1)
plt.axvline(x = mu, color = 'red', ls = "--", lw = 1, alpha = 1, label = "μ = " + str(round(mu, 1)))
plt.axvline(x = mu + sigma, color = 'orange', ls = "--", lw = 1, alpha = 1, label = "σ = " + str(np.round(sigma, 1)))
plt.axvline(x = mu - sigma, color = 'orange', ls = "--", lw = 1, alpha = 1)
plt.legend()
plt.savefig("histo_gaussian.png", dpi=600)
plt.show()

threshhold = 1*(mu + (2*sigma))
print("threshhold = " + str(threshhold))

mask_list = []

for i in raw_list:
    if i <= threshhold:
        i = 0
        mask_list.append(i)
    else:
        i = 255
        mask_list.append(i)

mask = np.array(mask_list).reshape(len(raw),len(raw[0]))

fsize_scale=2
fsize=(6.4*fsize_scale,4.8*fsize_scale)

plt.figure(figsize=(fsize))
plt.title('Raw Image')
plt.imshow(raw, cmap='gray')
plt.savefig("raw.png", dpi=600)
plt.show()

plt.figure(figsize=(fsize))
plt.title('Binary (2σ)')
plt.imshow(mask, cmap='gray')
plt.savefig("binary2sig.png", dpi=600)
plt.show()

raw_cropped = np.copy(raw)[150:4461, 150:2320]
mask_cropped = np.copy(mask)[150:4461, 150:2320]

mask_cropped[2070:2210, 700:810] = 0
mask_cropped[3055:3265, 566:685] = 0
mask_cropped[2750:3400, 1000:1610] = 0
mask_cropped[2545:2690, 760:890] = 0
mask_cropped[:, 1250:1320] = 0
mask_cropped[150:200, 950:1530] = 0
mask_cropped[274:295, 950:1500] = 0
mask_cropped[3055:3185, 2010:2160] = 0
mask_cropped[3055:3185, 2010:2160] = 0
mask_cropped[3920:3980, 380:450] = 0
mask_cropped[3545:3660, 1940:2020] = 0
mask_cropped[1220:1320, 1890:1983] = 0
mask_cropped[2120:2200, 1950:2010] = 0
mask_cropped[2100:2155, 2120:] = 0

mask_cropped_cv2 = mask_cropped.astype(np.uint8)
mask_cropped_cv2_lab = mask_cropped.astype(np.uint8)
cv2.imshow("mask_cropped_cv2_unlab.png", mask_cropped_cv2)
cv2.imwrite("mask_cropped_cv2_unlab.png", mask_cropped_cv2)

contours, _ = cv2.findContours(mask_cropped_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

thr_size = 500
box_scale = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > thr_size]
objects = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) < thr_size]

print('Number of objects = ' + str(len(objects)) + "\n")

for contour in contours:
    cv2.drawContours(mask_cropped_cv2_lab, contour, -1, (132, 1, 4), 3)

x_ds = []
y_ds = []

def put_labels(image, objects, box_scale):
    global x_ds, y_ds
    for i,box in enumerate(objects):
        x = box[0]
        y = max(0,box[1]-10)
        x_d =box[2] / box_scale[0][2] * 60 #multiply relative diameter by 100 (arbitrary)
        y_d =box[3] / box_scale[0][2] * 60
        x_ds.append(x_d)
        y_ds.append(y_d)
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        print('Object ' + str(i) + ': x-diameter: ' + '{:.2f}'.format(x_d) +
              ', y-diameter: ' +  '{:.2f}'.format(y_d))
        print(x, y, x_d, y_d)
        print('')

put_labels(mask_cropped_cv2_lab, objects, box_scale)

cv2.imshow('mask_cropped_lab',  cv2.resize(mask_cropped_cv2_lab, dsize=(0, 0), fx=0.5, fy=0.5))
cv2.imwrite("mask_cropped_cv2_lab.png", cv2.resize(mask_cropped_cv2_lab, dsize=(0, 0), fx=0.5, fy=0.5))

raw_masked = raw_cropped * mask_cropped

plt.figure(figsize=(fsize))
plt.title('Raw Masked')
plt.imshow(raw_masked, cmap='gray')
plt.savefig("raw_masked.png", dpi=600)
plt.show()

raw_masked_cv2 = raw_cropped * mask_cropped_cv2

plt.figure(figsize=(fsize))
plt.title('Raw Masked CV2')
plt.imshow(raw_masked_cv2, cmap='gray')
plt.savefig("raw_masked_cv2.png", dpi=600)
plt.show()

raw_masked_lab = raw_cropped * mask_cropped_cv2_lab

plt.figure(figsize=(fsize))
plt.title('Raw Masked CV2 Labelled')
plt.imshow(raw_masked_lab, cmap='gray')
plt.savefig("raw_masked_cv2_lab.png", dpi=600)
plt.show()

plt.figure(figsize=(fsize))
plt.title('Mask')
plt.imshow(mask_cropped, cmap='gray')
plt.savefig("mask_cropped.png", dpi=600)
plt.show()

def box_maker(Contour):

    """Generates list of all coords inside a contour """

    Contour = Contour[:, 0, :]
    x_min = np.min(Contour[:,0])
    x_max = np.max(Contour[:,0])
    y_min = np.min(Contour[:,1])
    y_max = np.max(Contour[:,1])

    x_coords = [x for x in range(x_min, x_max + 1)]
    y_coords = [y for y in range(y_min, y_max + 1)]

    output = [(x, y) for x in x_coords for y in y_coords]

    return output

def pixel_summer(pix_coords, image):

    """Takes in output from box_maker, finds all pixel values in image and return them summed"""

    contours_pixels = []
    for i in pix_coords:
        pix_val = image[i[1]][i[0]]
        contours_pixels.append(pix_val)
    summed = np.sum(contours_pixels)

    return summed

cat = []

num = 0

for C in contours:
    if C.shape[0] >= 0:
        BOX = box_maker(C)
        SUMMED = pixel_summer(BOX, raw_masked)
        cat.append(SUMMED)
    #else:
     #   summed_pixels.append(0)
    print(num, end='\r')
    num = num + 1 #just a counter to see how many countours are left

mags = []

for i in cat:
    m =  (2.53*10)-2.5*np.log10(i)
    mags.append(m)

cumvals = []

k = np.arange(2, 10.5, 0.1)

for i in k:
    cumval = sum(mags < i)
    cumvals.append(cumval)

y1 = np.log10(cumvals)
x1a = [k]
x1b = np.ravel(x1a)

m, b = np.polyfit(x1b, y1, 1)

print(m)
print(b)

fsize_scale=1
fsize=(6.4*fsize_scale,4.8*fsize_scale)

plt.figure(figsize=(fsize))
plt.title('Galaxy Number Counts')
plt.xlabel('Magnitude m')
plt.ylabel('log(Number Counts)')
plt.scatter(x1a, cumvals, s=5)
plt.yscale('log')
plt.plot(x1b, 10**(m*x1b + b), 'r')
plt.savefig("final_plot.png", dpi=600)
plt.show()
