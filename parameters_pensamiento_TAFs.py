# 1) Separate pdf pages: "separate_pdf.py"

# IMPORTANT
BLACKLIST = [3] # Add the question numbers you want to ignore here

# Set the maximum number of CPU cores to use for parallel processing
MAX_CORES = 30
# ---------------------


# Specific parameters of the exam

# Enable debugging mode. Will print intensities for each response, and the output_corners image
DEBUG = True  

# Folder containing input images
input_subfolder = 'Pensamiento/2026-Pensamiento-TAF'

# Parameter for Threshold Adjustment
THRESHOLD_ADJUSTMENT = 5  # Fine-tune the threshold (can be positive or negative)

# Exam parameters
num_questions = 6
options_QUESTIONS = ['A', 'B', 'C', 'D']
options_length = len(options_QUESTIONS)

# Scoring Parameters
correct_point = 1.0       # Points for a correct answer
incorrect_point = -0.33  # Points subtracted for an incorrect answer


# Parameters
# exec(open('py/parameters_global_old.py').read()) # Old size
exec(open('py/parameters_global.py').read())

# RUN correction
exec(open('py/read_exams.py').read())
