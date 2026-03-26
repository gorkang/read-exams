# Global parameters of the template

import os


# Response boxes positions ----------------------------------------------------

# --- Configuration for two-column layout ---
# Define the number of questions that fit in the first column.
# If num_questions (total) is greater than this, a second column is assumed.
NUM_QUESTIONS_FIRST_COLUMN = 30
# --- End Configuration ---

# SEGUNDA COLUMNA QUESTIONS para cuando hay mas de 33 preguntas
x_offset_start_ratio_QUESTIONS2 = 0.596 # DONDE EMPIEZAN EJE x
num_questions2 = num_questions - 33# ES  num_questions  33
# ------------------------------

# Start of boxes (ratios based on image dimensions)
x_offset_start_ratio_GRUPO = 0.75
x_offset_start_ratio_QUESTIONS_DNI = 0.222
x_offset_start_ratio_FORMA = 0.845

y_offset_start_ratio_DNI_GRUPO_FORMA = 0.049
y_offset_start_ratio_QUESTIONS = 0.312

# Size of boxes (ratios based on image dimensions)
box_size_x_ratio_all = 0.033
box_size_y_ratio_all = 0.0099

# Space between boxes (ratios based on image dimensions)
x_spacing_ratio = 0.062
y_spacing_ratio = 0.0197


# Other parameters -------------------------------------------------------

# Output folder
outputs_folder = 'outputs'               # Base folder for outputs
overwrite_output_images = True           # Whether to overwrite existing output images

# Reference image dimensions (width x height)
REFERENCE_WIDTH = 800
REFERENCE_HEIGHT = 1100

# Base font parameters for the reference image
BASE_FONT_SCALE = 0.35
BASE_THICKNESS = 1

# Corner detection
base_min_corner_circle_diameter = 30  # Minimum Corner Circle Diameter at Reference Size in pixels
corner_proximity_threshold = 0.25        # Threshold to detect corners based on image size
min_area_threshold_ratio = 0.00003       # Minimum area ratio to consider a contour as a corner


# Automatic parameters
input_folder = os.path.join("inputs", input_subfolder)
outputs_folder = os.path.join(outputs_folder, input_subfolder)
show_all_intensities = DEBUG             # Whether to display bubble intensities on images
output_images_folder = os.path.join(outputs_folder, 'output_images')          # Folder to save output images
output_corners_folder = os.path.join(outputs_folder, 'output_corners')  # Folder to save images with detected corners

# Define allowed image extensions
# allowed_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
allowed_image_extensions = ('.png')