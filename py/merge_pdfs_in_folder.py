import fitz
import os

def merge_pdfs_in_folder(folder_path):
    """
    Merges all PDF files in a given folder into a single in-memory PDF.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        fitz.Document or None: A fitz.Document object containing the merged PDF, or None if an error occurs.
    """

    pdf_files = []
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(folder_path, filename))
    except FileNotFoundError:
        print(f"Error: Folder not found: {folder_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return None

    merged_pdf = fitz.open()  # Create an empty PDF document

    try:
        for pdf_file in pdf_files:
            with fitz.open(pdf_file) as current_pdf:
                merged_pdf.insert_pdf(current_pdf)
        return merged_pdf

    except Exception as e:
        print(f"Error merging PDFs: {e}")
        return None


# folder_path = "inputs/Pensamiento/"  # Replace with your folder path
# merged_pdf = merge_pdfs_in_folder(folder_path)
