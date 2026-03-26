# IF THERE IS MORE THAN 1 TEMPLATE, RETURNS A DICTIONARY OF DICTIONARIES.
# READ_EXAMS THEN USES THE TEMPLATE BASED ON THE EXAM FORMA


import os
import cv2
import numpy as np
from skimage.filters import threshold_otsu

# Assuming these are defined elsewhere:
# is_image_file, detect_and_correct_page_orientation, detect_bubbles, plot_histogram,
# x_offset_start_ratio_QUESTIONS_DNI, y_offset_start_ratio_QUESTIONS, box_size_x_ratio_all,
# box_size_y_ratio_all, x_spacing_ratio, y_spacing_ratio, num_questions, options_length,
# options_QUESTIONS, show_all_intensities, x_offset_start_ratio_FORMA,
# y_offset_start_ratio_DNI_GRUPO_FORMA, THRESHOLD_ADJUSTMENT, output_corners_folder, output_folder

def correct_responses_template(input_folder):
    """
    Processes multiple template images and returns a nested dictionary of correct answers,
    keyed by the FORMA value found in each template.
    """

    all_correct_answers = {}  # Nested dictionary to store all results

    template_images = list(filter(lambda k: 'CORRECT_RESPONSES' in k.upper(), os.listdir(input_folder)))

    if not template_images:
        print("Error: No template images 'correct_responses' found in the input folder.")
        return None

    for img_filename in template_images:
        if is_image_file(img_filename):
            template_path = os.path.join(input_folder, img_filename)
            template_img = cv2.imread(template_path)
            if template_img is None:
                print(f"Error: Template image {img_filename} cannot be opened.")
                continue

            gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            _, binary_template = cv2.threshold(gray_template, 50, 255, cv2.THRESH_BINARY_INV)

            # Detect and correct page orientation for template
            corrected_template_img, reoriented = detect_and_correct_page_orientation(template_img, gray_template, binary_template, img_filename, output_corners_folder)
            gray_template = cv2.cvtColor(corrected_template_img, cv2.COLOR_BGR2GRAY)

            if reoriented:
                sections_params = {
                    'QUESTIONS': {
                        'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS_DNI,
                        'y_offset_start_ratio': y_offset_start_ratio_QUESTIONS,
                        'box_size_x_ratio': box_size_x_ratio_all,
                        'box_size_y_ratio': box_size_y_ratio_all,
                        'x_spacing_ratio': x_spacing_ratio,
                        'y_spacing_ratio': y_spacing_ratio,
                        'num_rows': num_questions,
                        'num_cols': options_length,
                        'options': options_QUESTIONS,
                        'direction': 'row-wise',
                        'show_all_intensities': show_all_intensities,
                        'validate_single_selection': False
                    },
                    'FORMA': {
                        'x_offset_start_ratio': x_offset_start_ratio_FORMA,
                        'y_offset_start_ratio': y_offset_start_ratio_DNI_GRUPO_FORMA,
                        'box_size_x_ratio': box_size_x_ratio_all,
                        'box_size_y_ratio': box_size_y_ratio_all,
                        'x_spacing_ratio': x_spacing_ratio,
                        'y_spacing_ratio': y_spacing_ratio,
                        'num_rows': 10,
                        'num_cols': 2,
                        'options': [str(i % 10) for i in range(1, 11)],
                        'direction': 'column-wise',
                        'show_all_intensities': show_all_intensities,
                        'validate_single_selection': True
                    }
                }

            # Add parameters for the second question column if needed
            if num_questions > NUM_QUESTIONS_FIRST_COLUMN:
                num_questions_col2 = num_questions - NUM_QUESTIONS_FIRST_COLUMN
                sections_params['QUESTIONS2'] = {
                                'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS2, # Ensure this is defined in global params
                                'y_offset_start_ratio': y_offset_start_ratio_QUESTIONS, # Assuming same vertical start as first column
                                'box_size_x_ratio': box_size_x_ratio_all,
                                'box_size_y_ratio': box_size_y_ratio_all,
                                'x_spacing_ratio': x_spacing_ratio,
                                'y_spacing_ratio': y_spacing_ratio,
                                'num_rows': num_questions_col2, # Number of questions in the second column
                                'num_cols': options_length,
                                'options': options_QUESTIONS,
                                'direction': 'row-wise',
                                'show_all_intensities': show_all_intensities,
                                'validate_single_selection': False
                            }


            # First call to detect_bubbles to collect intensities (current logic bases this on 'QUESTIONS' only)
            _, _, bubble_intensities = detect_bubbles(
                gray_template, corrected_template_img, sections_params['QUESTIONS'], 'QUESTIONS',
                global_threshold=0
            )

            # Convert intensities to NumPy array
            intensity_array = np.array(bubble_intensities, dtype=np.uint8)
            # Calculate the global threshold using threshold_otsu
            global_threshold_template = threshold_otsu(intensity_array, nbins=30) # Assuming nbins=30
            # Apply threshold adjustment
            adjusted_global_threshold_template = global_threshold_template + THRESHOLD_ADJUSTMENT # Assuming THRESHOLD_ADJUSTMENT is defined

            plot_histogram(
                dni=f"CORRECT_RESPONSES_TEMPLATE_{os.path.splitext(img_filename)[0]}", # Made DNI in plot unique per template file
                intensities=bubble_intensities,
                adjusted_threshold=adjusted_global_threshold_template,
                output_folder=outputs_folder # Ensure 'outputs_folder' is accessible here
            )

            marked_responses_questions, _, _ = detect_bubbles(
                gray_template, corrected_template_img, sections_params['QUESTIONS'], 'QUESTIONS',
                global_threshold=global_threshold_template,
                threshold_adjustment=THRESHOLD_ADJUSTMENT
            )

            marked_responses_forma, _, _ = detect_bubbles(
                gray_template, corrected_template_img, sections_params['FORMA'], 'FORMA',
                global_threshold=global_threshold_template,
                threshold_adjustment=THRESHOLD_ADJUSTMENT
            )

            # --- START: Added logic for QUESTIONS2 ---
            marked_responses_questions2 = {}
            if num_questions > NUM_QUESTIONS_FIRST_COLUMN and 'QUESTIONS2' in sections_params:
                marked_responses_questions2, _, _ = detect_bubbles(
                    gray_template, corrected_template_img, sections_params['QUESTIONS2'], 'QUESTIONS2',
                    global_threshold=global_threshold_template,
                    threshold_adjustment=THRESHOLD_ADJUSTMENT
                )
            # --- END: Added logic for QUESTIONS2 ---

            # Assuming `corrected_template_img` is the final processed overlay of the template
            output_template_image_path = os.path.join(output_images_folder, f"{img_filename}") # Ensure 'output_images_folder' is accessible
            cv2.imwrite(output_template_image_path, corrected_template_img)
            print(f"Template overlay saved as {output_template_image_path}")

            # --- START: Modified logic for extracting correct answers ---
            # Combine question responses from both columns (if applicable)
            marked_responses_all_questions_template = {}

            # Responses from the first column (QUESTIONS)
            for q_num, resp_list in marked_responses_questions.items():
                # q_num here is local to the 'QUESTIONS' section (e.g., 1 to NUM_QUESTIONS_FIRST_COLUMN)
                marked_responses_all_questions_template[q_num] = resp_list
            
            # Responses from the second column (QUESTIONS2), if it exists
            if num_questions > NUM_QUESTIONS_FIRST_COLUMN and marked_responses_questions2:
                for q_local, resp_list in marked_responses_questions2.items():
                    # q_local here is local to the 'QUESTIONS2' section (e.g., 1 to num_questions_col2)
                    # Convert to global question number
                    global_q_num = q_local + NUM_QUESTIONS_FIRST_COLUMN 
                    marked_responses_all_questions_template[global_q_num] = resp_list
            
            # Extract correct answers using combined responses
            correct_answers = {}
            # Iterate from 1 up to the total number of questions (e.g., 60, defined by global 'num_questions')
            for question_num_global in range(1, num_questions + 1): 
                response_list_for_q = marked_responses_all_questions_template.get(question_num_global, [])
                if response_list_for_q: # Template should ideally have one marked answer
                    correct_answers[question_num_global] = response_list_for_q[0]
                else:
                    correct_answers[question_num_global] = None # No answer marked for this question in the template
            # --- END: Modified logic for extracting correct answers ---

            # Process FORMA response (existing logic should be fine)
            forma_response = ''
            forma_responses_map = marked_responses_forma # Use the fetched forma responses
            for col in range(1, sections_params['FORMA']['num_cols'] + 1): # Iterate through columns of FORMA
                responses_in_col = forma_responses_map.get(col, [])
                if len(responses_in_col) == 1:
                    forma_response += responses_in_col[0]  # Only one bubble marked
                else:
                    forma_response += 'X'  # Multiple or no bubbles marked, or error

            correct_answers["FORMA"] = forma_response
            all_correct_answers[forma_response] = correct_answers
            print(f"Template image {img_filename} processed successfully.")
            
    return all_correct_answers
