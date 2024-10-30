import fitz  # PyMuPDF # sudo apt-get install python3-fitz
from PIL import Image
import os

# Define input PDF file and output folder
input_pdf = 'exams/IMPRIME_PSI P4 C3520-033_0060_001.pdf'
output_folder = 'exams'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the PDF file
pdf_document = fitz.open(input_pdf)

# Iterate over each page in the PDF
for page_num in range(pdf_document.page_count):
    # Select the page
    page = pdf_document[page_num]

    # Define output path for the PNG image
    output_path = os.path.join(output_folder, f'page_{page_num + 1}.png')

    # Render page as a PNG image with 300 DPI for high quality
    pix = page.get_pixmap(dpi=300)
    
    # Save the initial image
    pix.save(output_path)

    # Open the saved image to check orientation
    with Image.open(output_path) as img:
        width, height = img.size
        if width > height:
            # Rotate the image to portrait if it is in landscape
            img = img.rotate(90, expand=True)
            img.save(output_path)
            print(f"Rotated page {page_num + 1} to portrait orientation.")
        else:
            print(f"Page {page_num + 1} is already in portrait orientation.")

# Close the PDF document
pdf_document.close()
print("PDF extraction and orientation adjustment complete!")
