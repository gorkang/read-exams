
# Import libraries and functions -------------------------------

import cv2 # sudo apt-get install python3-opencv
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu  # sudo apt-get install python3-skimage


# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(allowed_image_extensions)

# Function to generate a unique filename by adding a timestamp if needed
def generate_unique_filename(output_folder, dni_value, extension=".png"):
    """
    Generates a unique filename by appending a timestamp if the filename already exists.
    
    Parameters:
        output_folder (str): Directory to save the image.
        dni_value (str): DNI value or placeholder.
        extension (str): File extension.
        
    Returns:
        str: Unique file path.
    """
    base_filename = os.path.join(output_folder, f"{dni_value}{extension}")
    if not os.path.exists(base_filename):
        return base_filename
    # Append a timestamp to ensure uniqueness
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    new_filename = os.path.join(output_folder, f"{dni_value}_{timestamp}{extension}")
    return new_filename

# Function to calculate scaling factors
def calculate_scaling_factors(image_width, image_height):
    """
    Calculates scaling factors based on image dimensions compared to reference dimensions.
    
    Parameters:
        image_width (int): Width of the current image.
        image_height (int): Height of the current image.
        
    Returns:
        font_scale (float): Scaled font size.
        thickness (int): Scaled text thickness.
    """
    width_scale = image_width / REFERENCE_WIDTH
    height_scale = image_height / REFERENCE_HEIGHT
    # Choose the minimum scale to maintain aspect ratio and prevent excessive scaling
    scaling_factor = min(width_scale, height_scale)

    font_scale = BASE_FONT_SCALE * scaling_factor
    # Ensure thickness is at least 1
    thickness = max(1, int(BASE_THICKNESS * scaling_factor))

    return font_scale, thickness

# Function to detect corners and reorient the page
def detect_and_correct_page_orientation(img, gray_img, binary_img, input_filename, output_folder):
    height, width = gray_img.shape

    # Calculate scaling factor based on image dimensions relative to reference
    width_scale = width / REFERENCE_WIDTH
    height_scale = height / REFERENCE_HEIGHT
    scaling_factor = min(width_scale, height_scale)

    # Adjust min_corner_circle_diameter based on scaling factor
    min_corner_circle_diameter = base_min_corner_circle_diameter * scaling_factor
    min_radius = int(min_corner_circle_diameter / 2)

    # Set maximum radius for circle detection (optional)
    max_corner_circle_diameter = base_min_corner_circle_diameter * 1.5 * scaling_factor
    max_radius = int(max_corner_circle_diameter / 2)

    # Detect circles using Hough Circle Transform
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,               # Inverse ratio of resolution
        minDist=100,          # Minimum distance between detected centers
        param1=50,            # Upper threshold for Canny edge detector
        param2=30,            # Threshold for center detection
        minRadius=min_radius,         # Minimum circle radius
        maxRadius=max_radius         # Maximum circle radius (adjust if needed)
    )

    # If no circles are detected, return early
    if circles is None:
        print("No circles detected.")
        return img, False

    # Convert circles to integer coordinates
    circles = np.round(circles[0, :]).astype("int")

    # No need to filter circles by diameter since minRadius already accounts for it

    # Define corner points for filtering
    corner_points = {
        'top-left': (0, 0),
        'top-right': (width, 0),
        'bottom-left': (0, height),
        'bottom-right': (width, height)
    }

    # Filter circles by finding the closest one to each corner
    closest_circles = {key: None for key in corner_points.keys()}
    closest_distances = {key: float('inf') for key in corner_points.keys()}

    for (x, y, r) in circles:
        for corner_name, (cx, cy) in corner_points.items():
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < closest_distances[corner_name]:
                closest_circles[corner_name] = (x, y, r)
                closest_distances[corner_name] = dist

    # Extract detected circles and check if we have all 4 corners
    detected_corners = [circle for circle in closest_circles.values() if circle is not None]
    if len(detected_corners) != 4:
        print("Could not detect all four corners.")
        return img, False

    # Draw detected circles on the image for visualization and save it
    if DEBUG:
        visual_img = img.copy()
        for (x, y, r) in detected_corners:
            cv2.circle(visual_img, (x, y), r, (0, 255, 0), 4)     # Green circle around the detected circle
            cv2.circle(visual_img, (x, y), 5, (255, 0, 0), -1)    # Small red dot at the center

        # Save the visual image with detected circles overlay
        output_path = os.path.join(output_folder, f"detected_corners_{os.path.basename(input_filename)}")
        cv2.imwrite(output_path, visual_img)
        print(f"Saved image with detected corners overlay: {output_path}")

    # Proceed to page orientation correction if all 4 circles are detected
    src_pts = np.array([
        closest_circles['top-left'][:2], 
        closest_circles['top-right'][:2], 
        closest_circles['bottom-right'][:2], 
        closest_circles['bottom-left'][:2]
    ], dtype='float32')

    # Calculate width and height for the destination points
    widthA = np.linalg.norm(np.array(src_pts[2]) - np.array(src_pts[3]))
    widthB = np.linalg.norm(np.array(src_pts[1]) - np.array(src_pts[0]))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(np.array(src_pts[1]) - np.array(src_pts[2]))
    heightB = np.linalg.norm(np.array(src_pts[0]) - np.array(src_pts[3]))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the perspective transform
    dst_pts = np.array([
        [0, 0], 
        [maxWidth - 1, 0], 
        [maxWidth - 1, maxHeight - 1], 
        [0, maxHeight - 1]
    ], dtype='float32')

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return corrected_img, True


# Function to detect bubbles in a given section using global threshold
def detect_bubbles(gray_img, corrected_img, params, section_name, global_threshold, threshold_adjustment=0):
    x_scale = gray_img.shape[1]
    y_scale = gray_img.shape[0]

    x_offset_start = params['x_offset_start_ratio'] * x_scale
    y_offset_start = params['y_offset_start_ratio'] * y_scale
    x_spacing = params['x_spacing_ratio'] * x_scale
    y_spacing = params['y_spacing_ratio'] * y_scale
    box_size_x = params['box_size_x_ratio'] * x_scale
    box_size_y = params['box_size_y_ratio'] * y_scale

    marked_responses = {}
    response_positions = {}
    bubble_intensities = []  # Collect intensities of all bubbles

    if params['direction'] == 'row-wise':
        for row in range(1, params['num_rows'] + 1):
            y_start = y_offset_start + (row - 1) * y_spacing
            marked_responses[row] = []
            response_positions[row] = {}

            row_bubble_intensities = []

            for col in range(1, params['num_cols'] + 1):
                x_start = x_offset_start + (col - 1) * x_spacing
                bubble_area = gray_img[int(y_start):int(y_start + box_size_y),
                                       int(x_start):int(x_start + box_size_x)]
                avg_intensity = np.mean(bubble_area)
                bubble_intensities.append(avg_intensity)  # Collect intensity
                row_bubble_intensities.append(avg_intensity)

                # Always draw the rectangle, regardless of whether the bubble is filled
                color = (200, 200, 200)
                cv2.rectangle(corrected_img, (int(x_start), int(y_start)),
                              (int(x_start + box_size_x), int(y_start + box_size_y)), color, 2)

                # Show intensities in all the boxes QUESTIONS
                if params['show_all_intensities']:
                    font_scale, thickness = calculate_scaling_factors(gray_img.shape[1], gray_img.shape[0])
                    intensity_text = f"{avg_intensity:.0f}"
                    text_position = (int(x_start), int(y_start - 5))
                    cv2.putText(corrected_img, intensity_text, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Mark responses based on global threshold
            selected_options = []
            adjusted_threshold = global_threshold + threshold_adjustment
            for i, avg_intensity in enumerate(row_bubble_intensities):
                if avg_intensity < adjusted_threshold:
                    option = params['options'][i]
                    selected_options.append(option)
                    x_start_current = x_offset_start + i * x_spacing
                    y_center = int(y_start + box_size_y / 2)
                    x_center = int(x_start_current + box_size_x / 2)
                    response_positions[row][option] = (x_center, y_center)
                    cv2.rectangle(corrected_img, (int(x_start_current), int(y_start)),
                                  (int(x_start_current + box_size_x), int(y_start + box_size_y)), (150, 50, 255), 2) # Selected responses
            marked_responses[row].extend(selected_options)
    else:  # 'column-wise'
        for col in range(1, params['num_cols'] + 1):
            x_start = x_offset_start + (col - 1) * x_spacing
            marked_responses[col] = []
            response_positions[col] = {}

            col_bubble_intensities = []

            for row in range(1, params['num_rows'] + 1):
                y_start = y_offset_start + (row - 1) * y_spacing
                bubble_area = gray_img[int(y_start):int(y_start + box_size_y),
                                       int(x_start):int(x_start + box_size_x)]
                avg_intensity = np.mean(bubble_area)
                bubble_intensities.append(avg_intensity)  # Collect intensity
                col_bubble_intensities.append(avg_intensity)

                # Always draw the rectangle, regardless of whether the bubble is filled
                color = (200, 200, 200)
                cv2.rectangle(corrected_img, (int(x_start), int(y_start)),
                              (int(x_start + box_size_x), int(y_start + box_size_y)), color, 2)

                # Show intensity in all the boxes DNI, GRUPO, FORMA
                if params['show_all_intensities']:
                    font_scale, thickness = calculate_scaling_factors(gray_img.shape[1], gray_img.shape[0])
                    intensity_text = f"{avg_intensity:.0f}"
                    text_position = (int(x_start), int(y_start - 5))
                    cv2.putText(corrected_img, intensity_text, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Mark responses based on global threshold
            selected_options = []
            adjusted_threshold = global_threshold + threshold_adjustment
            for i, avg_intensity in enumerate(col_bubble_intensities):
                if avg_intensity < adjusted_threshold:
                    option = params['options'][i]
                    selected_options.append(option)
                    y_start_current = y_offset_start + i * y_spacing
                    y_center = int(y_start_current + box_size_y / 2)
                    x_center = int(x_start + box_size_x / 2)
                    response_positions[col][option] = (x_center, y_center)
                    cv2.rectangle(corrected_img, (int(x_start), int(y_start_current)),
                                  (int(x_start + box_size_x), int(y_start_current + box_size_y)), (150, 50, 255), 2) # Selected responses

            marked_responses[col].extend(selected_options)

    # Validate single selection if required
    if params.get('validate_single_selection', False) and params['direction'] == 'column-wise':
        for col in marked_responses:
            if len(marked_responses[col]) > 1:
                # Mark each duplicate response with an 'X'
                options = marked_responses[col]
                for option in options:
                    position = response_positions[col].get(option, None)
                    if position:
                        # Draw the 'X' on the image
                        cv2.line(corrected_img, (position[0] - 10, position[1] - 10), 
                                 (position[0] + 10, position[1] + 10), (0, 0, 255), 2)
                        cv2.line(corrected_img, (position[0] - 10, position[1] + 10), 
                                 (position[0] + 10, position[1] - 10), (0, 0, 255), 2)
    return marked_responses, response_positions, bubble_intensities  # Return intensities

# Function to overlay symbols based on correctness
def overlay_symbols(corrected_img, response_positions, user_responses, correct_answers, section_direction):
    if section_direction == 'row-wise' or section_direction == 'column-wise':
        for key, user_response in user_responses.items():
            correct_answer = correct_answers.get(key, None)
            if not user_response:
                continue  # No response to mark
            if section_direction == 'row-wise':
                if len(user_response) > 1:
                    # Multiple responses - mark each with 'X's
                    for option in user_response:
                        position = response_positions.get(key, {}).get(option, None)
                        if position:
                            # Draw a red X
                            cv2.line(corrected_img, (position[0] - 10, position[1] - 10), 
                                     (position[0] + 10, position[1] + 10), (0, 0, 255), 2)
                            cv2.line(corrected_img, (position[0] - 10, position[1] + 10), 
                                     (position[0] + 10, position[1] - 10), (0, 0, 255), 2)
                else:
                    user_option = user_response[0]  # Assuming single response
                    correct_option = correct_answers.get(key, None)
                    if user_option == correct_option:
                        # Draw green circle
                        position = response_positions.get(key, {}).get(user_option, None)
                        if position:
                            # Adjust circle radius based on image size
                            image_height, image_width = corrected_img.shape[:2]
                            font_scale, thickness = calculate_scaling_factors(image_width, image_height)
                            radius = max(5, int(10 * (image_height / REFERENCE_HEIGHT)))  # Minimum radius of 5
                            cv2.circle(corrected_img, position, radius, (0, 255, 0), thickness)
                    else:
                        # Draw red X
                        position = response_positions.get(key, {}).get(user_option, None)
                        if position:
                            # Adjust line length based on image size
                            image_height, image_width = corrected_img.shape[:2]
                            scale = min(image_width, image_height) / max(REFERENCE_WIDTH, REFERENCE_HEIGHT)
                            line_length = max(5, int(10 * scale))
                            cv2.line(corrected_img, (position[0] - line_length, position[1] - line_length), 
                                     (position[0] + line_length, position[1] + line_length), (0, 0, 255), thickness=2)
                            cv2.line(corrected_img, (position[0] - line_length, position[1] + line_length), 
                                     (position[0] + line_length, position[1] - line_length), (0, 0, 255), thickness=2)
            elif section_direction == 'column-wise':
                if len(user_response) > 1:
                    # Multiple responses - mark each with 'X'
                    for option in user_response:
                        position = response_positions.get(key, {}).get(option, None)
                        if position:
                            # Draw a red X
                            cv2.line(corrected_img, (position[0] - 10, position[1] - 10), 
                                     (position[0] + 10, position[1] + 10), (0, 0, 255), 2)
                            cv2.line(corrected_img, (position[0] - 10, position[1] + 10), 
                                     (position[0] + 10, position[1] - 10), (0, 0, 255), 2)
                else:
                    user_option = user_response[0]  # Assuming single response
                    correct_option = correct_answers.get(key, None)
                    if user_option == correct_option:
                        # Draw green circle
                        position = response_positions.get(key, {}).get(user_option, None)
                        if position:
                            # Adjust circle radius based on image size
                            image_height, image_width = corrected_img.shape[:2]
                            font_scale, thickness = calculate_scaling_factors(image_width, image_height)
                            radius = max(5, int(10 * (image_height / REFERENCE_HEIGHT)))  # Minimum radius of 5
                            cv2.circle(corrected_img, position, radius, (0, 255, 0), thickness)
                    else:
                        # Draw red X
                        position = response_positions.get(key, {}).get(user_option, None)
                        if position:
                            # Adjust line length based on image size
                            image_height, image_width = corrected_img.shape[:2]
                            scale = min(image_width, image_height) / max(REFERENCE_WIDTH, REFERENCE_HEIGHT)
                            line_length = max(5, int(10 * scale))
                            cv2.line(corrected_img, (position[0] - line_length, position[1] - line_length), 
                                     (position[0] + line_length, position[1] + line_length), (0, 0, 255), thickness=2)
                            cv2.line(corrected_img, (position[0] - line_length, position[1] + line_length), 
                                     (position[0] + line_length, position[1] - line_length), (0, 0, 255), thickness=2)
    return corrected_img

# Function to add grade summary overlay on the image
def add_grade_overlay(corrected_img, num_correct, num_errors, num_no_responses, final_grade, image_width, image_height):
    """
    Adds a grade summary overlay on the image with dynamic text scaling.
    
    Parameters:
        corrected_img (numpy.ndarray): The corrected color image.
        num_correct (int): Number of correct responses.
        num_errors (int): Number of incorrect responses.
        num_no_responses (int): Number of unanswered questions.
        final_grade (float): Calculated final grade.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        
    Returns:
        corrected_img (numpy.ndarray): Image with grade summary overlay.
    """
    # Calculate scaling factors
    font_scale, thickness = calculate_scaling_factors(image_width, image_height)
    
    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    # Prepare the text parts and their colors
    texts = [
        ("Correcto: ", (0, 255, 0)),      # Green
        (f"{num_correct}", (0, 255, 0)),
        (" | Incorrecto: ", (0, 0, 255)), # Red
        (f"{num_errors}", (0, 0, 255)),
        (" | No respuesta: ", (128, 128, 128)),    # Grey
        (f"{num_no_responses}", (128, 128, 128)),
        (" | Nota: ", (0, 0, 0)),       # Black
        (f"{final_grade}/10", (0, 0, 0))
    ]

    # Starting position
    y_position = int(15 * (image_height / REFERENCE_HEIGHT))  # Adjust y_position based on scaling
    x_position = int(50 * (image_width / REFERENCE_WIDTH))   # Adjust x_position based on scaling

    # Draw each part sequentially
    for i in range(0, len(texts), 2):
        label, label_color = texts[i]
        value, value_color = texts[i+1]

        # Put the label
        cv2.putText(corrected_img, label, (x_position, y_position),
                    font, font_scale, label_color, thickness, line_type)
        # Calculate the width of the label to position the value next
        (label_width, _), _ = cv2.getTextSize(label, font, font_scale, thickness)
        # Position for the value
        cv2.putText(corrected_img, value, (x_position + label_width, y_position),
                    font, font_scale, value_color, thickness, line_type)
        # Update x_position for next part, adding some spacing for " | "
        spacing = int(cv2.getTextSize(" | ", font, font_scale, thickness)[0][0] * (image_width / REFERENCE_WIDTH))
        value_width = int(cv2.getTextSize(value, font, font_scale, thickness)[0][0] * (image_width / REFERENCE_WIDTH))
        x_position += label_width + value_width + spacing

    return corrected_img

# Function to highlight empty rows
def highlight_empty_rows(corrected_img, empty_rows, params, image_width, image_height):
    """
    Highlights unmarked bubbles in orange by overlaying semi-transparent rectangles.
    
    Parameters:
        corrected_img (numpy.ndarray): The corrected color image.
        empty_rows (list): List of row numbers with no responses.
        params (dict): Parameters defining the QUESTIONS section layout.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    
    Returns:
        corrected_img (numpy.ndarray): Image with highlighted bubbles.
    """
    overlay = corrected_img.copy()
    alpha = 0.2  # Transparency factor

    font_scale, thickness = calculate_scaling_factors(image_width, image_height)
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    img_height, img_width = corrected_img.shape[:2]
    y_offset_start = params['y_offset_start_ratio'] * img_height
    y_spacing = params['y_spacing_ratio'] * img_height
    box_size_x = params['box_size_x_ratio'] * img_width
    box_size_y = params['box_size_y_ratio'] * img_height
    x_spacing = params['x_spacing_ratio'] * img_width

    for row in empty_rows:
        y_start = int(y_offset_start + (row - 1) * y_spacing)
        for col in range(1, params['num_cols'] + 1):
            x_start = int(params['x_offset_start_ratio'] * img_width + (col - 1) * x_spacing)
            # Define the bubble area
            top_left = (x_start, y_start)
            bottom_right = (x_start + int(box_size_x), y_start + int(box_size_y))
            # Draw semi-transparent orange rectangle over the bubble
            cv2.rectangle(overlay, top_left, bottom_right, (0, 165, 255), -1)  # Orange in BGR

            # Optional: Add text indicating no response
            text = ""  # Text on empty rectangle overlay
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x_start + int((box_size_x - text_size[0]) / 2)
            text_y = y_start + int((box_size_y + text_size[1]) / 2)
            cv2.putText(overlay, text, (text_x, text_y), 
                        font, font_scale, (255, 255, 255), thickness, line_type)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, corrected_img, 1 - alpha, 0, corrected_img)
    return corrected_img

# Function to plot histograms for each student
def plot_histogram(dni, intensities, adjusted_threshold, output_folder):
    """
    Plots and saves the histogram of bubble intensities for a student.

    Parameters:
        dni (str): DNI of the student.
        intensities (list): List of bubble intensities.
        adjusted_threshold (float): Adjusted threshold used for bubble detection.
        output_folder (str): Directory to save the histogram image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(intensities, bins=50, color='blue', edgecolor='black')
    plt.axvline(x=adjusted_threshold, color='red', linestyle='--', label=f'Threshold: {adjusted_threshold}')
    plt.title(f'Histogram of Bubble Intensities (DNI: {dni})')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()

    # Set x-axis ticks to 10 evenly spaced values
    x_min, x_max = min(intensities), max(intensities)
    plt.xticks(np.linspace(x_min, x_max, 10))

    histogram_path = os.path.join(outputs_folder, f'output_intensities/{dni}_intensities.png')
    plt.savefig(histogram_path)
    plt.close()
    print(f"Saved histogram for DNI {dni} at {histogram_path}")
