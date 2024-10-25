# PARAMETERS --------------------------------------------------

input_folder = 'template2'  # Folder containing the images to process
output_folder = 'output_images'       # Folder where the overlay images will be saved
show_all_intensities = True  # Set to False to hide all intensities in the overlay for each response
overwrite_output_images = True  # Set to True to allow overwriting output images
intensity_divider = 1

# row_threshold = mean_intensity - (std_intensity / intensity_divider)


# Common parameters
corner_proximity_threshold = 0.25  # Keep flexibility for all corners
min_area_threshold_ratio = 0.00003  # Ratio to determine min area threshold


import cv2
import numpy as np
import pandas as pd
import os

# Function to generate a unique filename by adding a suffix if needed
def generate_unique_filename(output_folder, dni_value, extension=".png"):
    base_filename = os.path.join(output_folder, f"{dni_value}{extension}")
    if not os.path.exists(base_filename):
        return base_filename

    # If the file exists, add a suffix (e.g., _2, _3, etc.)
    counter = 2
    while True:
        new_filename = os.path.join(output_folder, f"{dni_value}_{counter}{extension}")
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

# Function to detect bubbles in a given section
def detect_bubbles(gray_img, corrected_img, params, section_name):
    x_scale = gray_img.shape[1]
    y_scale = gray_img.shape[0]

    x_offset_start = params['x_offset_start_ratio'] * x_scale
    y_offset_start = params['y_offset_start_ratio'] * y_scale
    x_spacing = params['x_spacing_ratio'] * x_scale
    y_spacing = params['y_spacing_ratio'] * y_scale
    box_size_x = params['box_size_x_ratio'] * x_scale
    box_size_y = params['box_size_y_ratio'] * y_scale

    marked_responses = {}
    bubble_intensities_all = []

    # Minimum intensity threshold to consider any bubble as marked
    empty_row_threshold = 200  # Adjust this value based on your images' intensity range

    if params['direction'] == 'row-wise':
        for row in range(1, params['num_rows'] + 1):
            y_start = y_offset_start + (row - 1) * y_spacing
            marked_responses[row] = []
            bubble_intensities = []

            for col in range(1, params['num_cols'] + 1):
                x_start = x_offset_start + (col - 1) * x_spacing
                bubble_area = gray_img[int(y_start):int(y_start + box_size_y),
                                       int(x_start):int(x_start + box_size_x)]
                avg_intensity = np.mean(bubble_area)
                bubble_intensities.append(avg_intensity)
                
                # Always draw the rectangle, regardless of whether the bubble is filled
                color = (0, 0, 255)
                cv2.rectangle(corrected_img, (int(x_start), int(y_start)),
                              (int(x_start + box_size_x), int(y_start + box_size_y)), color, 2)

                if params['show_all_intensities']:
                    intensity_text = f"{avg_intensity:.2f}"
                    text_position = (int(x_start), int(y_start - 5))
                    cv2.putText(corrected_img, intensity_text, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            # Calculate row-specific threshold
            mean_intensity = np.mean(bubble_intensities)
            std_intensity = np.std(bubble_intensities)
            row_threshold = mean_intensity - (std_intensity / 1.5)

            # Display the row threshold to the right of the last column for each row in QUESTIONS
            threshold_position = (int(x_start + x_spacing + box_size_x), int(y_start + box_size_y / 2))
            cv2.putText(corrected_img, f"Threshold: {row_threshold:.2f}", threshold_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  # Display in blue

            # Check if all bubbles in the row are above the empty row threshold
            if all(intensity > empty_row_threshold for intensity in bubble_intensities):
                continue  # Skip marking this row as it is empty

            # Mark responses based on row threshold
            for i, avg_intensity in enumerate(bubble_intensities):
                if avg_intensity < row_threshold:
                    option = params['options'][i]
                    marked_responses[row].append(option)
                    x_start = x_offset_start + i * x_spacing
                    cv2.rectangle(corrected_img, (int(x_start), int(y_start)),
                                  (int(x_start + box_size_x), int(y_start + box_size_y)), (0, 100, 0), 2)
    else:  # 'column-wise'
        for col in range(1, params['num_cols'] + 1):
            x_start = x_offset_start + (col - 1) * x_spacing
            marked_responses[col] = []
            bubble_intensities = []

            for row in range(1, params['num_rows'] + 1):
                y_start = y_offset_start + (row - 1) * y_spacing
                bubble_area = gray_img[int(y_start):int(y_start + box_size_y),
                                       int(x_start):int(x_start + box_size_x)]
                avg_intensity = np.mean(bubble_area)
                bubble_intensities.append(avg_intensity)

                # Always draw the rectangle, regardless of whether the bubble is filled
                color = (0, 0, 255)
                cv2.rectangle(corrected_img, (int(x_start), int(y_start)),
                              (int(x_start + box_size_x), int(y_start + box_size_y)), color, 2)

                if params['show_all_intensities']:
                    intensity_text = f"{avg_intensity:.2f}"
                    text_position = (int(x_start), int(y_start - 5))
                    cv2.putText(corrected_img, intensity_text, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            # Calculate column-specific threshold
            mean_intensity = np.mean(bubble_intensities)
            std_intensity = np.std(bubble_intensities)
            col_threshold = mean_intensity - (std_intensity / 2)

            # Display the column threshold at the bottom of each column for DNI, GRUPO, FORMA
            # Adjust y-position to be below the last row
            y_threshold_position = y_offset_start + params['num_rows'] * y_spacing
            threshold_position = (int(x_start), int(y_threshold_position + box_size_y / 2))
            cv2.putText(corrected_img, f"Threshold: {col_threshold:.2f}", threshold_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  # Display in blue

            # Check if all bubbles in the column are above the empty row threshold
            if all(intensity > empty_row_threshold for intensity in bubble_intensities):
                continue  # Skip marking this column as it is empty

            # Mark responses based on column threshold
            for i, avg_intensity in enumerate(bubble_intensities):
                if avg_intensity < col_threshold:
                    option = params['options'][i]
                    marked_responses[col].append(option)
                    y_start = y_offset_start + i * y_spacing
                    cv2.rectangle(corrected_img, (int(x_start), int(y_start)),
                                  (int(x_start + box_size_x), int(y_start + box_size_y)), (0, 100, 0), 2)

    return marked_responses



# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True) 

# Initialize an empty DataFrame to collect all results
results_df = pd.DataFrame()

# Iterate through all images in the input folder
for img_filename in os.listdir(input_folder):
    if not img_filename.endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    img_path = os.path.join(input_folder, img_filename)

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image {img_filename} not found or cannot be opened.")
        continue

    # Load the image and convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for improved contrast
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #gray_img = clahe.apply(gray_img)

    # Apply Gaussian blur to reduce noise
    #gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply adaptive thresholding for better separation of black and white regions
    _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)

    height, width = gray_img.shape
    detected_ellipses = []

    min_area_threshold = (width * height) * min_area_threshold_ratio

    # Circle detection modifications to handle distorted ellipses
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_threshold:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        if solidity < 0.85:
            continue

        # Calculate mean intensity inside the contour to ensure it's mostly black
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_intensity = cv2.mean(gray_img, mask=mask)[0]
        if mean_intensity > 50:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        proximity = False
        if cX < width * corner_proximity_threshold and cY < height * corner_proximity_threshold:
            corner = 'top-left'
            proximity = True
        elif cX > width * (1 - corner_proximity_threshold) and cY < height * corner_proximity_threshold:
            corner = 'top-right'
            proximity = True
        elif cX < width * corner_proximity_threshold and cY > height * (1 - corner_proximity_threshold):
            corner = 'bottom-left'
            proximity = True
        elif cX > width * (1 - corner_proximity_threshold) and cY > height * (1 - corner_proximity_threshold):
            corner = 'bottom-right'
            proximity = True

        if not proximity:
            continue

        if len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)
        detected_ellipses.append((corner, ellipse))

    # Ensure we have all 4 ellipses -------------
    if len(detected_ellipses) >= 4:
        ellipse_centers = {}
        for corner, ellipse in detected_ellipses:
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            ellipse_centers[corner] = center

        required_corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        if all(corner in ellipse_centers for corner in required_corners):
            top_left = ellipse_centers['top-left']
            top_right = ellipse_centers['top-right']
            bottom_left = ellipse_centers['bottom-left']
            bottom_right = ellipse_centers['bottom-right']

            src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
            width_A = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
            width_B = np.linalg.norm(np.array(top_right) - np.array(top_left))
            maxWidth = max(int(width_A), int(width_B))

            height_A = np.linalg.norm(np.array(top_right) - np.array(bottom_right))
            height_B = np.linalg.norm(np.array(top_left) - np.array(bottom_left))
            maxHeight = max(int(height_A), int(height_B))

            dst_pts = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype='float32')

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            corrected_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            gray_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)

            # Parameters for each section
            sections_params = {
                'QUESTIONS': {
                    'x_offset_start_ratio': 0.223,
                    'y_offset_start_ratio': 0.333,
                    'box_size_x_ratio': 0.029,
                    'box_size_y_ratio': 0.0095,
                    'x_spacing_ratio': 0.062,
                    'y_spacing_ratio': 0.0208,
                    'num_rows': 30,
                    'num_cols': 4,
                    'options': ['A', 'B', 'C', 'D'],
                    'direction': 'row-wise',
                    'show_all_intensities': show_all_intensities
                },
                'DNI': {
                    'x_offset_start_ratio': 0.223,
                    'y_offset_start_ratio': 0.0538,
                    'box_size_x_ratio': 0.029,
                    'box_size_y_ratio': 0.0095,
                    'x_spacing_ratio': 0.062,
                    'y_spacing_ratio': 0.0208,
                    'num_rows': 10,
                    'num_cols': 8,
                    'options': [str(i) for i in range(1, 11)],
                    'direction': 'column-wise',
                    'show_all_intensities': show_all_intensities
                },
                'GRUPO': {
                    'x_offset_start_ratio': 0.75,
                    'y_offset_start_ratio': 0.0538,
                    'box_size_x_ratio': 0.029,
                    'box_size_y_ratio': 0.0095,
                    'x_spacing_ratio': 0,
                    'y_spacing_ratio': 0.0208,
                    'num_rows': 10,
                    'num_cols': 1,
                    'options': [str(i if i < 10 else 0) for i in range(1, 11)],
                    'direction': 'column-wise',
                    'show_all_intensities': show_all_intensities
                },
                'FORMA': {
                    'x_offset_start_ratio': 0.845,
                    'y_offset_start_ratio': 0.0538,
                    'box_size_x_ratio': 0.029,
                    'box_size_y_ratio': 0.0095,
                    'x_spacing_ratio': 0.062,
                    'y_spacing_ratio': 0.0208,
                    'num_rows': 10,
                    'num_cols': 2,
                    'options': [str(i if i < 10 else 0) for i in range(1, 11)],
                    'direction': 'column-wise',
                    'show_all_intensities': show_all_intensities
                }
            }

            # Detect bubbles in each section
            marked_responses_questions = detect_bubbles(
                gray_img, corrected_img, sections_params['QUESTIONS'], 'QUESTIONS')

            marked_responses_DNI = detect_bubbles(
                gray_img, corrected_img, sections_params['DNI'], 'DNI')

            marked_responses_GRUPO = detect_bubbles(
                gray_img, corrected_img, sections_params['GRUPO'], 'GRUPO')

            marked_responses_FORMA = detect_bubbles(
                gray_img, corrected_img, sections_params['FORMA'], 'FORMA')

            # Process DNI response
            dni_value = ''
            for col in range(1, sections_params['DNI']['num_cols'] + 1):
                selected_options = marked_responses_DNI.get(col, [])
                if selected_options:
                    # Assuming only one selection per column
                    dni_digit = selected_options[0]
                    dni_value += dni_digit
                else:
                    dni_value += ' '  # Placeholder for missing digit

            # Process GRUPO response
            grupo_response = ''
            for col in range(1, sections_params['GRUPO']['num_cols'] + 1):
                selected_options = marked_responses_GRUPO.get(col, [])
                if selected_options:
                    grupo_response = selected_options[0]
                    break  # Only one bubble should be filled in GRUPO

            # Process FORMA response
            forma_response = ''
            for col in range(1, sections_params['FORMA']['num_cols'] + 1):
                selected_options = marked_responses_FORMA.get(col, [])
                if selected_options:
                    forma_response += selected_options[0]

            # Prepare data for the CSV row
            csv_data = {"DNI": dni_value, "GRUPO": grupo_response, "FORMA": forma_response, "image_name": img_filename}
            for question_num in range(1, 31):
                question_key = f"response{question_num:02}"
                marked_options = ','.join(marked_responses_questions.get(question_num, [])) if marked_responses_questions.get(question_num, []) else ''
                csv_data[question_key] = marked_options

            # Append the row to the results DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([csv_data])], ignore_index=True)
            # Generate a unique filename for the output image to avoid overwriting
            output_image_path = os.path.join(output_folder, f"{dni_value}.png") if overwrite_output_images else generate_unique_filename(output_folder, dni_value)
            cv2.imwrite(output_image_path, corrected_img)
            print(f"Overlay image saved as {output_image_path}")
        else:
            print(f"Not all four corners were detected in image {img_filename}.")
            # Append a row to results DataFrame indicating the error
            csv_data = {"DNI": "", "GRUPO": "", "FORMA": "", "image_name": img_filename}
            for question_num in range(1, 31):
                question_key = f"response{question_num:02}"
                csv_data[question_key] = ""
            results_df = pd.concat([results_df, pd.DataFrame([csv_data])], ignore_index=True)
    else:
        print(f"Less than 4 solid black ellipses detected in image {img_filename}.")

# Save the results to a CSV file
output_csv_filename = 'output.csv'
results_df.to_csv(output_csv_filename, index=False)
print(f"CSV file saved as {output_csv_filename}")
