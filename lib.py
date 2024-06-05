import cv2
import numpy as np
import matplotlib.pyplot as plt
import preprocess

def detect_template_shapes(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    reduced_color_image, centers = preprocess.reduce_colors(image_rgb, 8)
    hsv_image = cv2.cvtColor(reduced_color_image, cv2.COLOR_RGB2HSV)

    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([70, 70, 0])
    blue_upper = np.array([140, 255, 255])

    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    # Detect contours for red (half-moons) and blue (rectangle)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes = []

    image_height = image.shape[0]
    middle_y = image_height // 2
    
    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_x, centroid_y = 0, 0
        if centroid_y < middle_y:
            label = 'upper_red_half_moon'
        else:
            label = 'lower_red_half_moon'
        
        hu_moments = cv2.HuMoments(moments).flatten()
        detected_shapes.append((label, (x, y, w, h), hu_moments, (centroid_x, centroid_y)))
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for contour in contours_blue:
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_x, centroid_y = 0, 0

        hu_moments = cv2.HuMoments(moments).flatten()
        detected_shapes.append(('blue_rectangle', (x, y, w, h), hu_moments, (centroid_x, centroid_y)))
        
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)


    preprocess.display_image(image_rgb,'Detected Shapes')
    

    return detected_shapes

def detect_shapes(image_path, calculate_hu_moments=False):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_rgb = cv2.convertScaleAbs(image_rgb, alpha=2.0, beta=50)
    preprocess.display_image(image_rgb,'After processing')

    # Reduce colors to 8 clusters
    reduced_color_image, centers = preprocess.reduce_colors(image_rgb, 8)
    hsv_image = cv2.cvtColor(reduced_color_image, cv2.COLOR_RGB2HSV)

    # Define color ranges for red and blue in HSV
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([70, 70, 0])
    blue_upper = np.array([140, 255, 255])

    # Create masks for red and blue colors
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    # Detect contours for red (half-moons) and blue (rectangle)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes = []

    image_height = image.shape[0]
    middle_y = image_height // 2

    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_x, centroid_y = 0, 0
        if centroid_y < middle_y:
            label = 'upper_red_half_moon'
        else:
            label = 'lower_red_half_moon'
        if calculate_hu_moments:
            hu_moments = cv2.HuMoments(moments).flatten()
            detected_shapes.append((label, (x, y, w, h), hu_moments, (centroid_x, centroid_y)))
        else:
            detected_shapes.append((label, (x, y, w, h), (centroid_x, centroid_y)))
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for contour in contours_blue:
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_x, centroid_y = 0, 0
        if calculate_hu_moments:
            hu_moments = cv2.HuMoments(moments).flatten()
            detected_shapes.append(('blue_rectangle', (x, y, w, h), hu_moments, (centroid_x, centroid_y)))
        else:
            detected_shapes.append(('blue_rectangle', (x, y, w, h), (centroid_x, centroid_y)))
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title('Detected Shapes')
    plt.axis('off')
    plt.show()

    return detected_shapes