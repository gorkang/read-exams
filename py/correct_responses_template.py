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
                _, _, bubble_intensities = detect_bubbles(
                    gray_template, corrected_template_img, sections_params['QUESTIONS'], 'QUESTIONS',
                    global_threshold=0
                )

                intensity_array = np.array(bubble_intensities, dtype=np.uint8)
                global_threshold_template = threshold_otsu(intensity_array, nbins=30)
                adjusted_global_threshold_template = global_threshold_template + THRESHOLD_ADJUSTMENT

                plot_histogram(
                    dni="CORRECT_RESPONSES",
                    intensities=bubble_intensities,
                    adjusted_threshold=adjusted_global_threshold_template,
                    output_folder=output_folder
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

                output_template_image_path = os.path.join(output_folder, f"CORRECT_RESPONSES_{img_filename}.png")
                cv2.imwrite(output_template_image_path, corrected_template_img)
                print(f"Template overlay saved as {output_template_image_path}")

                correct_answers = {}
                for question_num in marked_responses_questions:
                    if marked_responses_questions[question_num]:
                        correct_answers[question_num] = marked_responses_questions[question_num][0]
                    else:
                        correct_answers[question_num] = None

                # Process FORMA response with validation
                forma_response = ''
                forma_responses = marked_responses_forma
                for col in range(1, sections_params['FORMA']['num_cols'] + 1):
                    responses = forma_responses.get(col, [])
                    if len(responses) == 1:
                        forma_response += responses[0]  # Only one bubble marked
                    else:
                        forma_response += 'X'  # Multiple or no bubbles marked

                
                #FORMA = marked_responses_forma[2][0] #ensure FORMA is a single value, not list.

                correct_answers["FORMA"] = forma_response

                all_correct_answers[forma_response] = correct_answers
                print(f"Template image {img_filename} processed successfully.")

    return all_correct_answers
