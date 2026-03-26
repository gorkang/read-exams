# TODO
# - Make this file a function. Two parameters: parameter files. The global parameters has a default value, the other one needs to be entered.
# - Change the instructions in README.md

import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


# Prevent OpenCV from competing with Python's multiprocessing for CPU threads
cv2.setNumThreads(0)

# PARAMETERS AND FUNCTIONS ----------------------------------

# Helper functions 
exec(open('py/helper_functions.py').read())
exec(open('py/correct_responses_template.py').read())
# exec(open('py/parameters_global.py').read()) 

def parse_section_response(marked_resp_map, num_cols):
    """Helper function to parse strings like DNI, GRUPO, FORMA."""
    val = ''
    for col in range(1, num_cols + 1):
        responses_list = marked_resp_map.get(col, [])
        val += responses_list[0] if len(responses_list) == 1 else 'X'
    return val

def process_single_exam(img_filename, correct_answers):
    """
    Worker function executed by each core. Processes exactly one exam image.
    """
    img_path = os.path.join(input_folder, img_filename)
    img = cv2.imread(img_path)
    
    # Base CSV template 
    csv_data = {
        "DNI": "X", "GRUPO": "", "FORMA": "",
        "image_name": img_filename,
        "number_of_questions": num_questions,
        "number_of_correct_responses": 0,
        "number_of_errors": 0,
        "number_of_no_responses": num_questions,
        "final_grade": 0.0,
        "processing_status": ""
    }
    # Pre-fill response columns with empty strings
    csv_data.update({f"response{q:02}": '' for q in range(1, num_questions + 1)})

    if img is None:
        csv_data["processing_status"] = "Image could not be opened"
        return csv_data, None

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
    
    corrected_img, reoriented = detect_and_correct_page_orientation(img, gray_img, binary_img, img_filename, output_corners_folder)
    gray_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)

    if not reoriented:
        csv_data["processing_status"] = "Corners not detected"
        return csv_data, None

    image_height, image_width = corrected_img.shape[:2]
    
    # Define parameters
    sections_params = {
        'DNI': {
            'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS_DNI,
            'y_offset_start_ratio': y_offset_start_ratio_DNI_GRUPO_FORMA,
            'box_size_x_ratio': box_size_x_ratio_all,
            'box_size_y_ratio': box_size_y_ratio_all,
            'x_spacing_ratio': x_spacing_ratio,
            'y_spacing_ratio': y_spacing_ratio,
            'num_rows': 10, 'num_cols': 8,
            'options': [str(i % 10) for i in range(1, 11)],
            'direction': 'column-wise',
            'show_all_intensities': show_all_intensities,
            'validate_single_selection': True
        },
        'QUESTIONS': { 
            'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS_DNI,
            'y_offset_start_ratio': y_offset_start_ratio_QUESTIONS,
            'box_size_x_ratio': box_size_x_ratio_all,
            'box_size_y_ratio': box_size_y_ratio_all,
            'x_spacing_ratio': x_spacing_ratio,
            'y_spacing_ratio': y_spacing_ratio,
            'num_rows': min(num_questions, NUM_QUESTIONS_FIRST_COLUMN),
            'num_cols': options_length,
            'options': options_QUESTIONS,
            'direction': 'row-wise',
            'show_all_intensities': show_all_intensities,
            'validate_single_selection': False
        },
        'GRUPO': {
            'x_offset_start_ratio': x_offset_start_ratio_GRUPO,
            'y_offset_start_ratio': y_offset_start_ratio_DNI_GRUPO_FORMA,
            'box_size_x_ratio': box_size_x_ratio_all,
            'box_size_y_ratio': box_size_y_ratio_all,
            'x_spacing_ratio': x_spacing_ratio,
            'y_spacing_ratio': y_spacing_ratio,
            'num_rows': 10, 'num_cols': 1,
            'options': [str(i % 10) for i in range(1, 11)],
            'direction': 'column-wise',
            'show_all_intensities': show_all_intensities,
            'validate_single_selection': True
        },
        'FORMA': {
            'x_offset_start_ratio': x_offset_start_ratio_FORMA,
            'y_offset_start_ratio': y_offset_start_ratio_DNI_GRUPO_FORMA,
            'box_size_x_ratio': box_size_x_ratio_all,
            'box_size_y_ratio': box_size_y_ratio_all,
            'x_spacing_ratio': x_spacing_ratio,
            'y_spacing_ratio': y_spacing_ratio,
            'num_rows': 10, 'num_cols': 2,
            'options': [str(i % 10) for i in range(1, 11)],
            'direction': 'column-wise',
            'show_all_intensities': show_all_intensities,
            'validate_single_selection': True
        }
    }

    if num_questions > NUM_QUESTIONS_FIRST_COLUMN:
        sections_params['QUESTIONS2'] = {
            'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS2,
            'y_offset_start_ratio': y_offset_start_ratio_QUESTIONS,
            'box_size_x_ratio': box_size_x_ratio_all,
            'box_size_y_ratio': box_size_y_ratio_all,
            'x_spacing_ratio': x_spacing_ratio,
            'y_spacing_ratio': y_spacing_ratio,
            'num_rows': num_questions - NUM_QUESTIONS_FIRST_COLUMN,
            'num_cols': options_length,
            'options': options_QUESTIONS,
            'direction': 'row-wise',
            'show_all_intensities': show_all_intensities,
            'validate_single_selection': False
        }
    
    # First pass: Collect all bubble intensities
    all_bubble_intensities = []
    marked_responses = {}
    response_positions = {}

    for section_name, params in sections_params.items():
        mr, rp, bubble_intensities_sec = detect_bubbles(
            gray_img, corrected_img, params, section_name, global_threshold=0
        )
        all_bubble_intensities.extend(bubble_intensities_sec)
        marked_responses[section_name] = mr
        response_positions[section_name] = rp

    # Calculate Otsu Threshold
    intensity_array = np.array(all_bubble_intensities, dtype=np.uint8)
    global_threshold = threshold_otsu(intensity_array)
    adjusted_global_threshold = global_threshold + THRESHOLD_ADJUSTMENT

    # Second pass: Re-process using adjusted threshold
    for section_name, params in sections_params.items():
        mr, rp, _ = detect_bubbles(
            gray_img, corrected_img, params, section_name,
            global_threshold=global_threshold,
            threshold_adjustment=THRESHOLD_ADJUSTMENT
        )
        marked_responses[section_name] = mr
        response_positions[section_name] = rp

    # Parse Identity Info
    dni_value = parse_section_response(marked_responses.get('DNI', {}), sections_params['DNI']['num_cols'])
    dni_clean = dni_value.strip() if dni_value.strip() else 'X'
    grupo_response = parse_section_response(marked_responses.get('GRUPO', {}), sections_params['GRUPO']['num_cols'])
    forma_response = parse_section_response(marked_responses.get('FORMA', {}), sections_params['FORMA']['num_cols'])

    histogram_data = {
        'dni': dni_clean,
        'intensities': all_bubble_intensities,
        'adjusted_threshold': adjusted_global_threshold
    }

    # Combine question responses
    marked_responses_all_questions = {}
    response_positions_all_questions = {}
    
    for q_num, resp_list in marked_responses.get('QUESTIONS', {}).items():
        marked_responses_all_questions[q_num] = resp_list
    for q_num, pos_data in response_positions.get('QUESTIONS', {}).items():
        response_positions_all_questions[q_num] = pos_data
    
    if 'QUESTIONS2' in marked_responses:
        for q_local, resp_list in marked_responses['QUESTIONS2'].items():
            marked_responses_all_questions[q_local + NUM_QUESTIONS_FIRST_COLUMN] = resp_list
        for q_local, pos_data in response_positions['QUESTIONS2'].items():
            response_positions_all_questions[q_local + NUM_QUESTIONS_FIRST_COLUMN] = pos_data

    # Grading Setup
    final_correct_answers = correct_answers.get(forma_response)
    if final_correct_answers is None:
        default_template = list(correct_answers.keys())[0] if correct_answers else None
        final_correct_answers = correct_answers.get(default_template, {}) if default_template else {}

    num_correct, num_errors, num_no_responses = 0, 0, 0
    empty_rows_global = []
    
    # Populate CSV dictionary and evaluate grades
    csv_data.update({
        "DNI": dni_clean,
        "GRUPO": grupo_response.strip() if grupo_response.strip() else 'X',
        "FORMA": forma_response.strip() if forma_response.strip() else 'X',
        "processing_status": "Success"
    })

    for question_num in range(1, num_questions + 1):
        user_response_list = marked_responses_all_questions.get(question_num, [])
        csv_data[f"response{question_num:02}"] = ','.join(user_response_list)
        correction_key = f"response{question_num:02}c"

        if question_num in BLACKLIST:
            csv_data[correction_key] = 'IGNORED'
            continue

        correct_answer = final_correct_answers.get(question_num)
        
        if not user_response_list:
            num_no_responses += 1
            empty_rows_global.append(question_num)
            csv_data[correction_key] = 0
        elif len(user_response_list) > 1:
            num_errors += 1
            csv_data[correction_key] = 0
        else:
            if user_response_list[0] == correct_answer:
                num_correct += 1
                csv_data[correction_key] = 1
            else:
                num_errors += 1
                csv_data[correction_key] = 0

    # Calculate Final Grade
    valid_questions_count = num_questions - len([q for q in BLACKLIST if q <= num_questions])
    final_grade_raw = (num_correct * correct_point) + (num_errors * incorrect_point)
    final_grade = max(0, min((final_grade_raw / valid_questions_count) * 10 if valid_questions_count > 0 else 0, 10))

    csv_data.update({
        "number_of_correct_responses": num_correct,
        "number_of_errors": num_errors,
        "number_of_no_responses": num_no_responses,
        "final_grade": round(final_grade, 2)
    })

    # Image Drawing/Highlighting
    empty_rows_col1 = [q for q in empty_rows_global if q <= NUM_QUESTIONS_FIRST_COLUMN]
    empty_rows_col2 = [q - NUM_QUESTIONS_FIRST_COLUMN for q in empty_rows_global if q > NUM_QUESTIONS_FIRST_COLUMN]
    
    if 'QUESTIONS' in sections_params:
        corrected_img = highlight_empty_rows(corrected_img, empty_rows_col1, sections_params['QUESTIONS'], image_width, image_height)
    if 'QUESTIONS2' in sections_params:
        corrected_img = highlight_empty_rows(corrected_img, empty_rows_col2, sections_params['QUESTIONS2'], image_width, image_height)

    primary_q_params = sections_params.get('QUESTIONS', sections_params.get('QUESTIONS2'))
    if primary_q_params:
        filtered_correct_answers = {k: v for k, v in final_correct_answers.items() if k not in BLACKLIST}
        corrected_img = overlay_symbols(
            corrected_img, response_positions_all_questions, marked_responses_all_questions,
            filtered_correct_answers, primary_q_params['direction']
        )

    corrected_img = add_grade_overlay(
        corrected_img, num_correct, num_errors, num_no_responses, round(final_grade, 2), image_width, image_height
    )

    resized_img = cv2.resize(corrected_img, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    output_image_path = os.path.join(output_images_folder, f"{dni_clean}.png") if overwrite_output_images else generate_unique_filename(output_images_folder, dni_clean)
    
    # Save the processed image directly from the worker
    cv2.imwrite(output_image_path, resized_img)

    return csv_data, histogram_data


def main():
    # Create folders
    intensities_path = os.path.join(outputs_folder, 'output_intensities/')
    os.makedirs(outputs_folder, exist_ok=True)
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(intensities_path, exist_ok=True)
    os.makedirs(output_corners_folder, exist_ok=True)

    # Get the correct template responses only ONCE
    correct_answers = correct_responses_template(input_folder=input_folder)

    # Compile a list of valid files to process
    files_to_process = [
        f for f in os.listdir(input_folder) 
        if not f.lower().startswith('correct_responses') and is_image_file(f)
    ]

    csv_records_list = []
    histogram_data_list = []

    print(f"Beginning processing of {len(files_to_process)} images across up to {MAX_CORES} cores...")

    # Spin up the multiprocessing pool
    with ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_exam, img_filename, correct_answers): img_filename 
            for img_filename in files_to_process
        }

        # Gather results as they finish 
        for future in as_completed(futures):
            img_filename = futures[future]
            try:
                csv_data, hist_data = future.result()
                if csv_data:
                    csv_records_list.append(csv_data)
                if hist_data:
                    histogram_data_list.append(hist_data)
                print(f"Finished processing: {img_filename}")
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")

    # Build and save final DataFrame once at the very end
    results_df = pd.DataFrame(csv_records_list)
    output_csv_filename = os.path.join(outputs_folder, 'output.csv')
    results_df.to_csv(output_csv_filename, index=False)
    print(f"CSV file saved successfully as {output_csv_filename}")

    # Plot the collected histograms sequentially at the end
    print("Plotting histograms...")
    for data in histogram_data_list:
        plot_histogram(
            dni=data['dni'], intensities=data['intensities'],
            adjusted_threshold=data['adjusted_threshold'], output_folder=outputs_folder
        )
    print("All tasks complete!")

# Mandatory for Windows/macOS multiprocessing to prevent recursive loop crashes
if __name__ == '__main__':
    main()