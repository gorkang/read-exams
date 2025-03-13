def correct_responses_template(input_folder):


    # Initialize correct answers dictionary
    correct_answers = {}

    # First, locate and process the TEMPLATE IMAGE image to extract correct answers
    template_found = False

    
    # list(filter(lambda x: x.startswith('correct_responses'), list(map(lambda x: x.lower(), os.listdir(input_folder)))))
    template_images = list(filter(lambda k: 'CORRECT_RESPONSES' in k, os.listdir(input_folder)))

    #for img_filename in os.listdir(input_folder):
    for img_filename in template_images:
        
        if img_filename.lower().startswith('correct_responses') and is_image_file(img_filename):
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
                image_height, image_width = corrected_template_img.shape[:2]
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
                # First call to detect_bubbles to collect intensities
                _, _, bubble_intensities = detect_bubbles(
                    gray_template, corrected_template_img, sections_params['QUESTIONS'], 'QUESTIONS',
                    global_threshold=0  # Placeholder, will compute next
                )
                
                # Convert intensities to NumPy array
                intensity_array = np.array(bubble_intensities, dtype=np.uint8)
                
                # Calculate the global threshold using threshold_otsu
                global_threshold_template = threshold_otsu(intensity_array, nbins=30)
                print(f"Global threshold (template): {global_threshold_template}")
                
                # Apply threshold adjustment
                adjusted_global_threshold_template = global_threshold_template + THRESHOLD_ADJUSTMENT

                # Plot histogram
                plot_histogram(
                        dni="CORRECT_RESPONSES",
                        intensities=bubble_intensities,
                        adjusted_threshold=adjusted_global_threshold_template,
                        output_folder=output_folder
                    )

                # Re-run detect_bubbles with the calculated global threshold
                marked_responses_questions, response_positions_questions, _ = detect_bubbles(
                    gray_template, corrected_template_img, sections_params['QUESTIONS'], 'QUESTIONS',
                    global_threshold=global_threshold_template,
                    threshold_adjustment=THRESHOLD_ADJUSTMENT
                )

                marked_responses_forma, response_positions_questions, _ = detect_bubbles(
                    gray_template, corrected_template_img, sections_params['FORMA'], 'FORMA',
                    global_threshold=global_threshold_template,
                    threshold_adjustment=THRESHOLD_ADJUSTMENT
                )

                # Assuming `corrected_template_img` is the final processed overlay of the template
                output_template_image_path = os.path.join(output_folder, "CORRECT_RESPONSES.png")
                cv2.imwrite(output_template_image_path, corrected_template_img)
                print(f"Template overlay saved as {output_template_image_path}")
                
                # Extract correct answers
                for question_num in marked_responses_questions:
                    if marked_responses_questions[question_num]:
                        correct_answers[question_num] = marked_responses_questions[question_num][0]
                    else:
                        correct_answers[question_num] = None  # No correct answer marked

                FORMA = marked_responses_forma[2]

                template_found = True
                print(f"Template image {img_filename} processed successfully.")

                # Add FORMA to correct_answers
                correct_answers["FORMA"] = FORMA

                return correct_answers
                break

    if not template_found:
        print("Error: Template image 'correct_responses.png' not found in the input folder or an error occurred processing it.")
        # Depending on requirements, you might want to exit or continue without grading
        exit(1)
