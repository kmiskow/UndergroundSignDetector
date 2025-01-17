import cv2
import numpy as np
from preprocess import display_image, reduce_colors, calculate_hu_moments, calculate_moments,apply_filter
class Detector:
    def __init__(self, template_path:str):
        self.template_shapes = self._detect_shapes(template_path)
    def detect(self,image_path:str, debug = False):
        test_shapes = self._detect_shapes(image_path,debug = debug)

        matched_shapes = self._match_shapes(self.template_shapes, test_shapes)

        image_test = cv2.imread(image_path)
        image_test_rgb = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
        for match in matched_shapes:
            test_shape, template_shape = match
            x, y, w, h = test_shape[1]
            cv2.rectangle(image_test_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if debug:
            display_image(image_test_rgb, "Image with matched rectangles")

        valid_signs = self._check_underground_sign_combinations(matched_shapes)
        if debug:
            print("Valid underground signs:", valid_signs)
        print(f"Detected {len(valid_signs)} valid signs" )


        image_test_rgb = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
        for sign in valid_signs:
            x_min, y_min, x_max, y_max, _ ,_= sign
            cv2.rectangle(image_test_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image_test_rgb, "Underground Sign", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        display_image(image_test_rgb,'Test Image with Valid Underground Signs')


    def _detect_shapes(self,image_path,debug = False):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.5, beta=0)
        kernel = np.array([[0.0625, 0.125, 0.0625],
                   [0.125, 0.25, 0.125],
                   [0.0625, 0.125, 0.0625]])
        image_rgb = apply_filter(image_rgb,kernel)

        reduced_color_image, _ = reduce_colors(image_rgb, 8)
        if debug:
            display_image(reduced_color_image, "image after k-means")
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

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_shapes = []
        image_height = image.shape[0]
        image_width = image.shape[1]
        middle_y = image_height // 2

        for contour in contours_red:
            x, y, w, h = cv2.boundingRect(contour)
            moments = calculate_moments(contour)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
            else:
                continue
            if center_y < y+h/2:
                label = 'upper_red_half_moon'
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 255, 0), 2)
            else:
                label = 'lower_red_half_moon'
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

            hu_moments = calculate_hu_moments(moments).flatten()
            detected_shapes.append((label, (x, y, w, h), hu_moments, (center_x, center_y)))


        for contour in contours_blue:
            x, y, w, h = cv2.boundingRect(contour)
            moments = calculate_moments(contour)
            hu_moments = calculate_hu_moments(moments).flatten()

            detected_shapes.append(('blue_rectangle', (x, y, w, h), hu_moments))
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if debug:
            display_image(image_rgb, title = 'Detected Shapes')

        return detected_shapes
    
    def _match_shapes(self,template_shapes, test_shapes, distance_threshold=0.5):
        matched_shapes = []

        for test_shape in test_shapes:
            best_match = None
            min_distance = float('inf')
            for template_shape in template_shapes:
                if test_shape[0] == template_shape[0]:
                    distance = np.linalg.norm(test_shape[2] - template_shape[2])
                    # print(f"Matching {test_shape[0]}: distance = {distance:.5f}")
                    if distance < min_distance:
                        min_distance = distance
                        best_match = template_shape
            if best_match and min_distance < distance_threshold:
                matched_shapes.append((test_shape, best_match))
                # print(f"Matched {test_shape[0]} with distance {min_distance:.5f}")

        return matched_shapes
    
    def _check_underground_sign_combinations(self,matched_shapes):
        upper_halfmoons = [shape for shape in matched_shapes if shape[0][0] == 'upper_red_half_moon']
        lower_halfmoons = [shape for shape in matched_shapes if shape[0][0] == 'lower_red_half_moon']
        blue_rectangles = [shape for shape in matched_shapes if shape[0][0] == 'blue_rectangle']

        possible_signs = []
        used = []
        for upper in upper_halfmoons:
            for lower in lower_halfmoons:
                upper_center_x, upper_center_y = upper[0][3]
                lower_center_x, lower_center_y = lower[0][3]
                distance_x = abs(upper_center_x - lower_center_x)
                distance_y = abs(upper_center_y - lower_center_y)
                if distance_x < 0.5 * upper[0][1][2]  and distance_y<2*upper[0][1][3] and upper[0][1][2] < lower[0][1][2] *1.1:
                    for blue in blue_rectangles:
                        if blue not in used:
                            upper_y = upper[0][1][1] + upper[0][1][3]
                            lower_y = lower[0][1][1]
                            blue_y = blue[0][1][1]
                            blue_height = blue[0][1][3]
                            if (upper_y - lower_y)/2 + lower_y > blue_y and (upper_y - lower_y)/2 + lower_y < blue_y+blue_height:
                                x_min = min(upper[0][1][0], lower[0][1][0], blue[0][1][0])
                                y_min = upper[0][1][1]
                                x_max = max(upper[0][1][0] + upper[0][1][2], lower[0][1][0] + lower[0][1][2], blue[0][1][0] + blue[0][1][2])
                                y_max = lower[0][1][1] + lower[0][1][3]
                                possible_signs.append((x_min, y_min, x_max, y_max, distance_x,distance_y))
                                used.append(blue)
                                break
                    else:
                        x_min = min(upper[0][1][0], lower[0][1][0])
                        y_min = upper[0][1][1]
                        x_max = max(upper[0][1][0] + upper[0][1][2], lower[0][1][0] + lower[0][1][2])
                        y_max = lower[0][1][1] + lower[0][1][3]
                        possible_signs.append((x_min, y_min, x_max, y_max, distance_x,distance_y))
        return possible_signs
