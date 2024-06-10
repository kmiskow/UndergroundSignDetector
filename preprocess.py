import matplotlib.pyplot as plt
import cv2
import numpy as np


def display_image(image,title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def reduce_colors(image, k):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))
    return result_image, center

def apply_filter(image, kernel):
    image_height, image_width, image_channels = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)

    output_image = np.zeros_like(image)

    for y in range(image_height):
        for x in range(image_width):
            for c in range(image_channels):
                region = padded_image[y:y + kernel_height, x:x + kernel_width, c]
                output_image[y, x, c] = np.sum(region * kernel)
    
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image


def calculate_moments(contour):
    moments = cv2.moments(contour)
    return moments
# def calculate_moments(contour):
#     contour_float32 = np.array(contour).astype(np.float32)

#     moments = {
#         'm00': np.float32(0),
#         'm10': np.float32(0),
#         'm01': np.float32(0),
#         'm20': np.float32(0),
#         'm11': np.float32(0),
#         'm02': np.float32(0),
#         'm30': np.float32(0),
#         'm21': np.float32(0),
#         'm12': np.float32(0),
#         'm03': np.float32(0)
#     }

#     if cv2.contourArea(contour_float32) > 0:
#         a00 = np.float32(0)
#         a10 = np.float32(0)
#         a01 = np.float32(0)
#         a20 = np.float32(0)
#         a11 = np.float32(0)
#         a02 = np.float32(0)
#         a30 = np.float32(0)
#         a21 = np.float32(0)
#         a12 = np.float32(0)
#         a03 = np.float32(0)
#         xi_1, yi_1 = contour_float32[0][0]
#         xi_12 = xi_1 * xi_1
#         yi_12 = yi_1 * yi_1

#         for point in contour_float32[1:]:
#             xi, yi = point[0]

#             xi2 = xi * xi
#             yi2 = yi * yi
#             dxy = xi_1 * yi - xi * yi_1
#             xii_1 = xi_1 + xi
#             yii_1 = yi_1 + yi

#             a00 += dxy
#             a10 += dxy * xii_1
#             a01 += dxy * yii_1
#             a20 += dxy * (xi_1 * xii_1 + xi2)
#             a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi))
#             a02 += dxy * (yi_1 * yii_1 + yi2)
#             a30 += dxy * xii_1 * (xi_12 + xi2)
#             a03 += dxy * yii_1 * (yi_12 + yi2)
#             a21 += dxy * (xi_12 * (3 * yi_1 + yi) + 2 * xi * xi_1 * yii_1 + xi2 * (yi_1 + 3 * yi))
#             a12 += dxy * (yi_12 * (3 * xi_1 + xi) + 2 * yi * yi_1 * xii_1 + yi2 * (xi_1 + 3 * xi))

#             xi_1, yi_1 = xi, yi
#             xi_12 = xi2
#             yi_12 = yi2

#         if np.abs(a00) > np.finfo(float).eps:
#             if a00 > 0:
#                 db1_2 = 0.5
#                 db1_6 = 0.16666666666666666666666666666667
#                 db1_12 = 0.083333333333333333333333333333333
#                 db1_24 = 0.041666666666666666666666666666667
#                 db1_20 = 0.05
#                 db1_60 = 0.016666666666666666666666666666667
#             else:
#                 db1_2 = -0.5
#                 db1_6 = -0.16666666666666666666666666666667
#                 db1_12 = -0.083333333333333333333333333333333
#                 db1_24 = -0.041666666666666666666666666666667
#                 db1_20 = -0.05
#                 db1_60 = -0.016666666666666666666666666666667

#             moments['m00'] = a00 * db1_2
#             moments['m10'] = a10 * db1_6
#             moments['m01'] = a01 * db1_6
#             moments['m20'] = a20 * db1_12
#             moments['m11'] = a11 * db1_24
#             moments['m02'] = a02 * db1_12
#             moments['m30'] = a30 * db1_20
#             moments['m21'] = a21 * db1_60
#             moments['m12'] = a12 * db1_60
#             moments['m03'] = a03 * db1_20

#     return moments

def calculate_central_moments(moments):
    if moments['m00'] == 0:
        return None
    x_bar = moments['m10'] / moments['m00']
    y_bar = moments['m01'] / moments['m00']
    
    mu = {}
    mu['mu20'] = moments['m20'] - x_bar * moments['m10']
    mu['mu02'] = moments['m02'] - y_bar * moments['m01']
    mu['mu11'] = moments['m11'] - x_bar * moments['m01']
    mu['mu30'] = moments['m30'] - 3 * x_bar * moments['m20'] + 2 * x_bar**2 * moments['m10']
    mu['mu03'] = moments['m03'] - 3 * y_bar * moments['m02'] + 2 * y_bar**2 * moments['m01']
    mu['mu21'] = moments['m21'] - 2 * x_bar * moments['m11'] - y_bar * moments['m20'] + 2 * x_bar**2 * moments['m01']
    mu['mu12'] = moments['m12'] - 2 * y_bar * moments['m11'] - x_bar * moments['m02'] + 2 * y_bar**2 * moments['m10']
    
    return mu

def calculate_normalized_moments(mu, moments):
    if mu is None or moments['m00'] == 0:
        return None
    nu = {}
    nu['nu20'] = mu['mu20'] / moments['m00']**2
    nu['nu02'] = mu['mu02'] / moments['m00']**2
    nu['nu11'] = mu['mu11'] / moments['m00']**2
    nu['nu30'] = mu['mu30'] / moments['m00']**(5/2)
    nu['nu03'] = mu['mu03'] / moments['m00']**(5/2)
    nu['nu21'] = mu['mu21'] / moments['m00']**(5/2)
    nu['nu12'] = mu['mu12'] / moments['m00']**(5/2)
    
    return nu

def calculate_hu_moments(moments):
    mu = calculate_central_moments(moments)
    nu = calculate_normalized_moments(mu, moments)
    hu = np.zeros(7)

    if nu is None:
        return hu
    
    hu[0] = nu['nu20'] + nu['nu02']
    hu[1] = (nu['nu20'] - nu['nu02'])**2 + 4*nu['nu11']**2
    hu[2] = (nu['nu30'] - 3*nu['nu12'])**2 + (3*nu['nu21'] - nu['nu03'])**2
    hu[3] = (nu['nu30'] + nu['nu12'])**2 + (nu['nu21'] + nu['nu03'])**2
    hu[4] = (nu['nu30'] - 3*nu['nu12'])*(nu['nu30'] + nu['nu12'])*((nu['nu30'] + nu['nu12'])**2 - 3*(nu['nu21'] + nu['nu03'])**2) + (3*nu['nu21'] - nu['nu03'])*(nu['nu21'] + nu['nu03'])*(3*(nu['nu30'] + nu['nu12'])**2 - (nu['nu21'] + nu['nu03'])**2)
    hu[5] = (nu['nu20'] - nu['nu02'])*((nu['nu30'] + nu['nu12'])**2 - (nu['nu21'] + nu['nu03'])**2) + 4*nu['nu11']*(nu['nu30'] + nu['nu12'])*(nu['nu21'] + nu['nu03'])
    hu[6] = (3*nu['nu21'] - nu['nu03'])*(nu['nu30'] + nu['nu12'])*((nu['nu30'] + nu['nu12'])**2 - 3*(nu['nu21'] + nu['nu03'])**2) - (nu['nu30'] - 3*nu['nu12'])*(nu['nu21'] + nu['nu03'])*(3*(nu['nu30'] + nu['nu12'])**2 - (nu['nu21'] + nu['nu03'])**2)
    
    return hu