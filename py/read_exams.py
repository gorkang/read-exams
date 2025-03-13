# TODO

# - Make this file a function. Two parameters: parameter files. The global parameters has a default value, the other one needs to be entered.
# - Change the instructions in README.md

# PARAMETERS AND FUNCTIONS ----------------------------------

# Helper functions
exec(open('py/helper_functions.py').read())
exec(open('py/correct_responses_template.py').read())


# Global parameters of the template
#exec(open('py/parameters_global.py').read())


# Create folders, etc. --------------------------------------------------------------------------------

# Create output folders if they don't exist
os.makedirs(outputs_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_corners_folder, exist_ok=True)
results_df = pd.DataFrame()

# List to store histogram data for each student
histogram_data_list = []


# TEMPLATE CORRECT RESPONSES -----------------------------------------------------------------------------
# Correct template
correct_answers = correct_responses_template(input_folder=input_folder)


# STUDENT RESPONSES -------------------------------------------------------------------------------------
#img_filename = os.listdir(input_folder)[4]
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
                "DNI": "X",  # Assign 'X' for processing failure
                "GRUPO": "", 
                "FORMA": "", 
                "image_name": img_filename,
                "number_of_questions": num_questions,
                "number_of_correct_responses": 0,
                "number_of_errors": 0,
                "number_of_no_responses": num_questions,
                "final_grade": 0,
                "processing_status": "Image could not be opened"
            }
            csv_data.update({
                f"response{question_num:02}": '' 
                for question_num in range(1, 31)
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
            sections_params = {
                'DNI': {
                    'x_offset_start_ratio': x_offset_start_ratio_QUESTIONS_DNI, 
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

            # First pass: Collect all bubble intensities
            all_bubble_intensities = []
            marked_responses = {}
            response_positions = {}

            # Process DNI first (but will re-extract DNI after second pass)
            mr_dni, rp_dni, bubble_intensities_dni = detect_bubbles(
                gray_img, corrected_img, sections_params['DNI'], 'DNI',
                global_threshold=0  # Placeholder
            )
            all_bubble_intensities.extend(bubble_intensities_dni)
            marked_responses['DNI'] = mr_dni
            response_positions['DNI'] = rp_dni

            # Process the rest of the sections
            for section_name in ['GRUPO', 'FORMA', 'QUESTIONS']:
                params = sections_params[section_name]
                mr, rp, bubble_intensities = detect_bubbles(
                    gray_img, corrected_img, params, section_name,
                    global_threshold=0  # Placeholder
                )
                all_bubble_intensities.extend(bubble_intensities)
                marked_responses[section_name] = mr
                response_positions[section_name] = rp


            # THRESHOLD CALCULATION ----------------------------------------------

            # Convert intensities to NumPy array
            intensity_array = np.array(all_bubble_intensities, dtype=np.uint8)

            # OTSU ---
            
            # Calculate the global threshold using threshold_otsu
            global_threshold = threshold_otsu(intensity_array)


            # Works well for empty pages. 
            #global_threshold = reference_intensity - reference_intensity * .2

            print(f"Global threshold for {img_filename}: {global_threshold}")

            # END THRESHOLD CALCULATION ----------------------------------------------


            # Apply threshold adjustment (only used for histogram)
            adjusted_global_threshold = global_threshold + THRESHOLD_ADJUSTMENT

            # Second pass: Re-process each section using the adjusted global threshold
            marked_responses = {}
            response_positions = {}
            for section_name, params in sections_params.items():
                mr, rp, _ = detect_bubbles(
                    gray_img, corrected_img, params, section_name,
                    global_threshold=global_threshold,
                    threshold_adjustment=THRESHOLD_ADJUSTMENT
                )
                marked_responses[section_name] = mr
                response_positions[section_name] = rp

            # Process DNI response with validation
            dni_value = ''
            dni_responses = marked_responses['DNI']
            for col in range(1, sections_params['DNI']['num_cols'] + 1):
                responses = dni_responses.get(col, [])
                if len(responses) == 1:
                    dni_value += responses[0]  # Only one bubble marked, take it
                else:
                    dni_value += 'X'  # No response or multiple responses detected


            dni_clean = dni_value.strip() if dni_value.strip() else 'X'  # For filenames

            # Collect histogram data (after correct DNI extraction)
            histogram_data_list.append({
                'dni': dni_clean,
                'intensities': all_bubble_intensities,
                'adjusted_threshold': adjusted_global_threshold
            })

            # Process GRUPO response with validation
            grupo_response = ''
            grupo_responses = marked_responses['GRUPO']
            for col in range(1, sections_params['GRUPO']['num_cols'] + 1):
                responses = grupo_responses.get(col, [])
                if len(responses) == 1:
                    grupo_response += responses[0]  # Only one bubble marked
                else:
                    grupo_response += 'X'  # Multiple or no bubbles marked

            # Process FORMA response with validation
            forma_response = ''
            forma_responses = marked_responses['FORMA']
            for col in range(1, sections_params['FORMA']['num_cols'] + 1):
                responses = forma_responses.get(col, [])
                if len(responses) == 1:
                    forma_response += responses[0]  # Only one bubble marked
                else:
                    forma_response += 'X'  # Multiple or no bubbles marked

            # Compare QUESTIONS responses with correct answers
            num_correct = 0
            num_errors = 0
            num_no_responses = 0
            empty_rows = []  # To store rows with no responses

            marked_responses_questions = marked_responses['QUESTIONS']
            response_positions_questions = response_positions['QUESTIONS']

            # CHOOSE TEMPLATE FOR CORRECTION
            final_correct_answers = correct_answers.get(forma_response)
            # If we cannot get one because forma does not match, gets one anyway
            # REVIEW: what happens when we have 2 templates but the person did not write a FORMA?
            if (final_correct_answers is None):
                for name,dict_ in correct_answers.items():
                    name_dict = name
                final_correct_answers = correct_answers.get(name_dict, None)
                

            for question_num in range(1, num_questions + 1):
                user_response = marked_responses_questions.get(question_num, [])
                correct_answer = final_correct_answers.get(question_num, None)
                if not user_response:
                    num_no_responses += 1
                    empty_rows.append(question_num)  # Assuming question_num corresponds to row number
                elif len(user_response) > 1:
                    # Multiple responses - all are errors, count as one error
                    num_errors += 1
                else:
                    if user_response[0] == correct_answer:
                        num_correct += 1
                    else:
                        num_errors += 1

            # Calculate final grade
            final_grade = (num_correct * correct_point) + (num_errors * incorrect_point)
            final_grade = (final_grade / num_questions) * 10
            final_grade = max(0, min(final_grade, 10))  # Ensure grade is between 0 and 10

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
            # Add responses and their correctness
            for question_num in range(1, num_questions + 1):
                response_key = f"response{question_num:02}"
                correction_key = f"response{question_num:02}c"

                user_response = marked_responses_questions.get(question_num, [])
                correct_answer = final_correct_answers.get(question_num, None)

                # Store response (join multiple responses with commas if needed)
                csv_data[response_key] = ','.join(user_response) if user_response else ''

                # Determine correctness
                if not user_response:
                    csv_data[correction_key] = 0  # No response = incorrect
                elif len(user_response) > 1:
                    csv_data[correction_key] = 0  # Multiple responses = incorrect
                else:
                    csv_data[correction_key] = 1 if user_response[0] == correct_answer else 0

            
            # Highlight empty rows in orange (only over bubble areas)
            corrected_img = highlight_empty_rows(
                corrected_img, 
                empty_rows, 
                sections_params['QUESTIONS'],
                image_width, 
                image_height
            )

            # Overlay symbols based on correctness
            corrected_img = overlay_symbols(
                corrected_img, 
                response_positions_questions, 
                marked_responses_questions, 
                final_correct_answers, 
                sections_params['QUESTIONS']['direction']
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

            # Downsample the final image to REFERENCE_WIDTH x REFERENCE_HEIGHT
            resized_img = cv2.resize(corrected_img, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)

            # Append to results dataframe
            results_df = pd.concat([results_df, pd.DataFrame([csv_data])], ignore_index=True)
            
            # Save overlay image
            dni_clean = dni_value.strip() if dni_value.strip() else 'X'  # Ensure correct dni_clean
            output_image_path = os.path.join(output_folder, f"{dni_clean}.png") if overwrite_output_images else generate_unique_filename(output_folder, dni_clean)
            cv2.imwrite(output_image_path, resized_img)  # Save the resized image
            print(f"Processed and saved overlay image for {img_filename} as {output_image_path}")
        else:
            # Handle images where corners are not detected
            # This else corresponds to 'if reoriented:' being False
            csv_data = {
                "DNI": "",  # You can choose to leave these empty or set to a default value
                "GRUPO": "", 
                "FORMA": "", 
                "image_name": img_filename,
                "number_of_questions": num_questions,
                "number_of_correct_responses": 0,
                "number_of_errors": 0,
                "number_of_no_responses": num_questions,
                "final_grade": 0,
                "processing_status": "Corners not detected"
            }
            csv_data.update({
                f"response{question_num:02}": '' 
                for question_num in range(1, 31)
            })
            results_df = pd.concat([results_df, pd.DataFrame([csv_data])], ignore_index=True)
            print(f"Corners not detected for {img_filename}. Added to CSV.")
            continue  # Skip further processing for this image

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
        output_folder=output_folder
    )
