# TODO

# - Make this file a function. Two parameters: parameter files. The global parameters has a default value, the other one needs to be entered.
# - Change the instructions in README.md

# PARAMETERS AND FUNCTIONS ----------------------------------

# Helper functions
exec(open('py/helper_functions.py').read())
exec(open('py/correct_responses_template.py').read())


# Global parameters of the template
# IMPORTANT: Ensure 'num_questions' is set in your global parameters (e.g., to 60).
# Also, ensure parameters for the second question column like 'x_offset_start_ratio_QUESTIONS2' are defined.
# exec(open('py/parameters_global.py').read()) # This line should load num_questions, etc.




# Create folders, etc. --------------------------------------------------------------------------------
intensities_path = os.path.join(outputs_folder, f'output_intensities/')

# Create output folders if they don't exist
os.makedirs(outputs_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(intensities_path, exist_ok=True)
os.makedirs(output_corners_folder, exist_ok=True)
results_df = pd.DataFrame()

# List to store histogram data for each student
histogram_data_list = []


# TEMPLATE CORRECT RESPONSES -----------------------------------------------------------------------------
# Correct template
correct_answers = correct_responses_template(input_folder=input_folder)


# STUDENT RESPONSES -------------------------------------------------------------------------------------
# Example: img_filename = os.listdir(input_folder)[4]
# Now, iterate through all images again to process student responses
for img_filename in os.listdir(input_folder):
    # Skip 'CORRECT_RESPONSES.png' explicitly
    if img_filename.lower().startswith('correct_responses'):
        continue

    if is_image_file(img_filename):
        img_path = os.path.join(input_folder, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Image {img_filename} not found or cannot be opened.")
            # Append to CSV with 'X' in DNI and mark as processing failed
            csv_data = {
                "DNI": "X",
                "GRUPO": "",
                "FORMA": "",
                "image_name": img_filename,
                "number_of_questions": num_questions, # Global num_questions
                "number_of_correct_responses": 0,
                "number_of_errors": 0,
                "number_of_no_responses": num_questions, # Global num_questions
                "final_grade": 0,
                "processing_status": "Image could not be opened"
            }
            # Ensure response columns are generated up to the total number of questions
            csv_data.update({
                f"response{question_num:02}": ''
                for question_num in range(1, num_questions + 1) # Use global num_questions
            })
            results_df = pd.concat([results_df, pd.DataFrame([csv_data])], ignore_index=True)
            print(f"Appended processing failure for {img_filename}.")
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Detect and correct page orientation
        corrected_img, reoriented = detect_and_correct_page_orientation(img, gray_img, binary_img, img_filename, output_corners_folder)

        gray_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)

        if reoriented:
            image_height, image_width = corrected_img.shape[:2]
            
            # Define parameters for all sections, including potentially two question columns
            sections_params = {
                'DNI': {
                    'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS_DNI, # Assuming this name is general or DNI specific
                    'y_offset_start_ratio': y_offset_start_ratio_DNI_GRUPO_FORMA,
                    'box_size_x_ratio': box_size_x_ratio_all,
                    'box_size_y_ratio': box_size_y_ratio_all,
                    'x_spacing_ratio': x_spacing_ratio,
                    'y_spacing_ratio': y_spacing_ratio,
                    'num_rows': 10,
                    'num_cols': 8,
                    'options': [str(i % 10) for i in range(1, 11)],
                    'direction': 'column-wise',
                    'show_all_intensities': show_all_intensities,
                    'validate_single_selection': True
                },
                'QUESTIONS': { # Parameters for the first column of questions
                    'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS_DNI, # Or a specific x_offset for Q1
                    'y_offset_start_ratio': y_offset_start_ratio_QUESTIONS,
                    'box_size_x_ratio': box_size_x_ratio_all,
                    'box_size_y_ratio': box_size_y_ratio_all,
                    'x_spacing_ratio': x_spacing_ratio,
                    'y_spacing_ratio': y_spacing_ratio,
                    'num_rows': min(num_questions, NUM_QUESTIONS_FIRST_COLUMN), # Rows for the first column
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
                    'num_rows': 10,
                    'num_cols': 1,
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
            
            # First pass: Collect all bubble intensities
            all_bubble_intensities = []
            marked_responses = {} # Stores responses per section { 'DNI': {...}, 'QUESTIONS': {...}, 'QUESTIONS2': {...} }
            response_positions = {} # Stores positions per section

            # Process DNI first
            mr_dni, rp_dni, bubble_intensities_dni = detect_bubbles(
                gray_img, corrected_img, sections_params['DNI'], 'DNI',
                global_threshold=0  # Placeholder for first pass
            )
            all_bubble_intensities.extend(bubble_intensities_dni)
            marked_responses['DNI'] = mr_dni
            response_positions['DNI'] = rp_dni

            # Define sections to process in the loop
            sections_to_process_loop = ['GRUPO', 'FORMA', 'QUESTIONS']
            if 'QUESTIONS2' in sections_params: # If second column exists
                sections_to_process_loop.append('QUESTIONS2')

            for section_name in sections_to_process_loop:
                params = sections_params[section_name]
                mr, rp, bubble_intensities_sec = detect_bubbles( # Changed variable name
                    gray_img, corrected_img, params, section_name,
                    global_threshold=0  # Placeholder for first pass
                )
                all_bubble_intensities.extend(bubble_intensities_sec)
                marked_responses[section_name] = mr
                response_positions[section_name] = rp

            # THRESHOLD CALCULATION ----------------------------------------------
            intensity_array = np.array(all_bubble_intensities, dtype=np.uint8)
            global_threshold = threshold_otsu(intensity_array)
            adjusted_global_threshold = global_threshold + THRESHOLD_ADJUSTMENT
            # END THRESHOLD CALCULATION ------------------------------------------

            # Second pass: Re-process each section using the adjusted global threshold
            # marked_responses and response_positions will be updated here
            for section_name, params in sections_params.items(): # Process all defined sections
                mr, rp, _ = detect_bubbles(
                    gray_img, corrected_img, params, section_name,
                    global_threshold=global_threshold,
                    threshold_adjustment=THRESHOLD_ADJUSTMENT
                )
                marked_responses[section_name] = mr
                response_positions[section_name] = rp

            # Process DNI response with validation
            dni_value = ''
            dni_responses_map = marked_responses.get('DNI', {}) # Use .get for safety
            for col in range(1, sections_params['DNI']['num_cols'] + 1):
                responses_list = dni_responses_map.get(col, [])
                if len(responses_list) == 1:
                    dni_value += responses_list[0]
                else:
                    dni_value += 'X'
            dni_clean = dni_value.strip() if dni_value.strip() else 'X'

            histogram_data_list.append({
                'dni': dni_clean,
                'intensities': all_bubble_intensities,
                'adjusted_threshold': adjusted_global_threshold
            })

            # Process GRUPO response
            grupo_response = ''
            grupo_responses_map = marked_responses.get('GRUPO', {})
            for col in range(1, sections_params['GRUPO']['num_cols'] + 1):
                responses_list = grupo_responses_map.get(col, [])
                if len(responses_list) == 1:
                    grupo_response += responses_list[0]
                else:
                    grupo_response += 'X'

            # Process FORMA response
            forma_response = ''
            forma_responses_map = marked_responses.get('FORMA', {})
            for col in range(1, sections_params['FORMA']['num_cols'] + 1):
                responses_list = forma_responses_map.get(col, [])
                if len(responses_list) == 1:
                    forma_response += responses_list[0]
                else:
                    forma_response += 'X'

            # --- Combine question responses from both columns (if applicable) ---
            marked_responses_all_questions = {}
            response_positions_all_questions = {}

            # Responses from the first column
            mr_col1 = marked_responses.get('QUESTIONS', {})
            rp_col1 = response_positions.get('QUESTIONS', {})
            for q_num, resp_list in mr_col1.items(): # q_num is 1-based, local to this column
                marked_responses_all_questions[q_num] = resp_list
            for q_num, pos_data in rp_col1.items():
                response_positions_all_questions[q_num] = pos_data
            
            # Responses from the second column (if it exists)
            if num_questions > NUM_QUESTIONS_FIRST_COLUMN and 'QUESTIONS2' in marked_responses:
                mr_col2 = marked_responses.get('QUESTIONS2', {})
                rp_col2 = response_positions.get('QUESTIONS2', {})
                for q_local, resp_list in mr_col2.items(): # q_local is 1-based for column 2
                    global_q_num = q_local + NUM_QUESTIONS_FIRST_COLUMN # Convert to global question number
                    marked_responses_all_questions[global_q_num] = resp_list
                for q_local, pos_data in rp_col2.items():
                    global_q_num = q_local + NUM_QUESTIONS_FIRST_COLUMN
                    response_positions_all_questions[global_q_num] = pos_data
            # --- End combining question responses ---

            # Compare ALL QUESTIONS responses with correct answers
            num_correct = 0
            num_errors = 0
            num_no_responses = 0
            empty_rows_global = []  # Stores global question numbers (1 up to num_questions)

            final_correct_answers = correct_answers.get(forma_response)
            if final_correct_answers is None: # Fallback if forma is not matched
                # This fallback picks the 'last' template if multiple exist and forma is invalid.
                # Consider if a default or error is more appropriate.
                default_template_name = list(correct_answers.keys())[0] if correct_answers else None
                if default_template_name:
                    final_correct_answers = correct_answers.get(default_template_name, {})
                else: # No templates loaded at all
                    final_correct_answers = {} 
                    print(f"Warning: No correct answer templates loaded for image {img_filename}.")


            for question_num in range(1, num_questions + 1): # Iterate 1 to total questions (e.g., 60)
                user_response_list = marked_responses_all_questions.get(question_num, [])
                # Ensure final_correct_answers is not None before using .get
                correct_answer = final_correct_answers.get(question_num, None) if final_correct_answers else None
                
                if not user_response_list:
                    num_no_responses += 1
                    empty_rows_global.append(question_num)
                elif len(user_response_list) > 1: # Multiple responses
                    num_errors += 1
                else: # Single response
                    if user_response_list[0] == correct_answer:
                        num_correct += 1
                    else:
                        num_errors += 1

            # Calculate final grade
            final_grade = (num_correct * correct_point) + (num_errors * incorrect_point)
            # Ensure num_questions is not zero to avoid division by zero if script is run with num_questions=0
            final_grade = (final_grade / num_questions) * 10 if num_questions > 0 else 0
            final_grade = max(0, min(final_grade, 10))

            # Prepare CSV data
            csv_data = {
                "DNI": dni_value if dni_value.strip() else 'X',
                "GRUPO": grupo_response if grupo_response.strip() else 'X',
                "FORMA": forma_response if forma_response.strip() else 'X',
                "image_name": img_filename,
                "number_of_questions": num_questions,
                "number_of_correct_responses": num_correct,
                "number_of_errors": num_errors,
                "number_of_no_responses": num_no_responses,
                "final_grade": round(final_grade, 2),
                "processing_status": "Success"
            }
            # Add all responses (up to num_questions) and their correctness to CSV
            for question_num in range(1, num_questions + 1):
                response_key = f"response{question_num:02}"
                correction_key = f"response{question_num:02}c"

                user_response_list = marked_responses_all_questions.get(question_num, [])
                # Ensure final_correct_answers is not None before using .get
                correct_answer = final_correct_answers.get(question_num, None) if final_correct_answers else None

                csv_data[response_key] = ','.join(user_response_list) if user_response_list else ''

                if not user_response_list:
                    csv_data[correction_key] = 0
                elif len(user_response_list) > 1:
                    csv_data[correction_key] = 0
                else:
                    csv_data[correction_key] = 1 if user_response_list[0] == correct_answer else 0

            # Highlight empty rows in orange (handles two columns)
            # Prepare lists of empty rows, local to each column section
            empty_rows_col1_local = []
            empty_rows_col2_local = []
            for q_global in empty_rows_global:
                if 1 <= q_global <= NUM_QUESTIONS_FIRST_COLUMN:
                    empty_rows_col1_local.append(q_global) # These are local row numbers for section QUESTIONS
                elif NUM_QUESTIONS_FIRST_COLUMN < q_global <= num_questions:
                    # Convert global question number to local row number for QUESTIONS2 section
                    empty_rows_col2_local.append(q_global - NUM_QUESTIONS_FIRST_COLUMN)
            
            # Highlight for the first column
            if 'QUESTIONS' in sections_params:
                corrected_img = highlight_empty_rows(
                    corrected_img,
                    empty_rows_col1_local,
                    sections_params['QUESTIONS'], # Parameters for the first column
                    image_width,
                    image_height
                )
            # Highlight for the second column (if it exists)
            if 'QUESTIONS2' in sections_params:
                corrected_img = highlight_empty_rows(
                    corrected_img,
                    empty_rows_col2_local,
                    sections_params['QUESTIONS2'], # Parameters for the second column
                    image_width,
                    image_height
                )

            # Overlay symbols based on correctness (using combined data)
            # This assumes overlay_symbols uses absolute coordinates from response_positions_all_questions
            # and the 'direction' parameter is general enough or primarily for context.
            # If overlay_symbols itself needs to know about distinct columns, it might need adjustment
            # or be called twice (once per column, with corresponding data subsets).
            # For now, we try with combined data.
            primary_question_section_params = sections_params.get('QUESTIONS', sections_params.get('QUESTIONS2')) # Get a representative direction
            if primary_question_section_params:
                 corrected_img = overlay_symbols(
                    corrected_img,
                    response_positions_all_questions, # Combined, global keys, absolute bubble positions
                    marked_responses_all_questions,   # Combined, global keys
                    final_correct_answers,            # Template with global keys
                    primary_question_section_params['direction'] # General direction
                )

            # Add grade summary overlay on the image
            corrected_img = add_grade_overlay(
                corrected_img,
                num_correct,
                num_errors,
                num_no_responses,
                round(final_grade, 2),
                image_width,
                image_height
            )

            resized_img = cv2.resize(corrected_img, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
            results_df = pd.concat([results_df, pd.DataFrame([csv_data])], ignore_index=True)
            
            output_image_path = os.path.join(output_images_folder, f"{dni_clean}.png") if overwrite_output_images else generate_unique_filename(output_images_folder, dni_clean)
            cv2.imwrite(output_image_path, resized_img)
            print(f"Processed and saved overlay image for {img_filename} as {output_image_path}")
        else:
            # Handle images where corners are not detected
            csv_data = {
                "DNI": "",
                "GRUPO": "",
                "FORMA": "",
                "image_name": img_filename,
                "number_of_questions": num_questions, # Global num_questions
                "number_of_correct_responses": 0,
                "number_of_errors": 0,
                "number_of_no_responses": num_questions, # Global num_questions
                "final_grade": 0,
                "processing_status": "Corners not detected"
            }
            # Ensure response columns are generated up to the total number of questions
            csv_data.update({
                f"response{question_num:02}": ''
                for question_num in range(1, num_questions + 1) # Use global num_questions
            })
            results_df = pd.concat([results_df, pd.DataFrame([csv_data])], ignore_index=True)
            print(f"Corners not detected for {img_filename}. Added to CSV.")
            continue

# Save the results to a CSV file
output_csv_filename = os.path.join(outputs_folder, 'output.csv')
results_df.to_csv(output_csv_filename, index=False)
print(f"CSV file saved as {output_csv_filename}")

# After processing all images, plot histograms for each student
for data in histogram_data_list:
    plot_histogram(
        dni=data['dni'],
        intensities=data['intensities'],
        adjusted_threshold=data['adjusted_threshold'],
        output_folder=outputs_folder # Assuming intensities_path was intended for individual files, not the plot output dir
    )