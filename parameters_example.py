# Specific parameters of the exam

# Enable debugging mode. Will print intensities for each response, and the output_corners image
DEBUG = True  

# Folder containing input images
input_subfolder = 'example'

# Parameter for Threshold Adjustment
THRESHOLD_ADJUSTMENT = 5  # Fine-tune the threshold (can be positive or negative)

# Exam parameters
num_questions = 30
options_QUESTIONS = ['A', 'B', 'C', 'D']
options_length = len(options_QUESTIONS)

# Scoring Parameters
correct_point = 1.0       # Points for a correct answer
incorrect_point = -0.33  # Points subtracted for an incorrect answer

# RUN correction
# Global parameters of the template
exec(open('py/parameters_global_old.py').read())
exec(open('py/read_exams.py').read())
