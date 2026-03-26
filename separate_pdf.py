import fitz  # PyMuPDF
from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION --------------------------------------

# Set the maximum number of CPU cores to use 
MAX_CORES = 30

# Define input PDF folder and output folder
input_folder = 'inputs/Pensamiento/2026-Pensamiento-TAF/'
# input_folder = 'inputs/Pensamiento/2026-Pensamiento-parcial1-tarde/'
output_folder = input_folder

# --------------------------------------------------------


# Load the external merge function
exec(open('py/merge_pdfs_in_folder.py').read())

def process_page(pdf_path, page_num, output_folder):
    """
    Worker function executed by each core. 
    Opens the PDF, extracts a specific page, and adjusts orientation.
    """
    # Open the document independently in this worker process
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    output_path = os.path.join(output_folder, f'exam_{page_num + 1}.png')
    
    # Render page as a PNG image with 300 DPI for high quality
    pix = page.get_pixmap(dpi=300)
    pix.save(output_path)
    
    # Close the document to free memory in this worker
    doc.close()
    
    # Open the saved image to check orientation
    with Image.open(output_path) as img:
        width, height = img.size
        if width > height:
            # Rotate the image to portrait if it is in landscape
            img = img.rotate(90, expand=True)
            img.save(output_path)
            return f"Rotated page {page_num + 1} to portrait orientation."
        else:
            return f"Page {page_num + 1} is already in portrait orientation."

def main():
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Creates a single pdf with all the individual pdfs
    print("Merging PDFs...")
    pdf_document = merge_pdfs_in_folder(input_folder)
    page_count = pdf_document.page_count

    # Save the merged document to a temporary file 
    # (Required because PyMuPDF objects cannot be passed between processes)
    temp_pdf_path = os.path.join(output_folder, 'temp_merged_document.pdf')
    pdf_document.save(temp_pdf_path)
    pdf_document.close()

    print(f"Processing {page_count} pages across up to {MAX_CORES} cores...")

    # Start the parallel processing pool
    with ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        # Submit all page tasks to the pool
        futures = {
            executor.submit(process_page, temp_pdf_path, page_num, output_folder): page_num
            for page_num in range(page_count)
        }

        # Print results as they finish (they will complete out-of-order)
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")

    # Clean up the temporary merged PDF
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    print("PDF extraction and orientation adjustment complete!")

# Protect the entry point (Required for multiprocessing in Python)
if __name__ == '__main__':
    main()
