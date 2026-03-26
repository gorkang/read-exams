# SETUP ----------------------------------------------------------

# Install python packages
# sudo apt install python3-venv python3-pip python3-dev python3-opencv python3-skimage python3-fitz


# Not needed in Positron. But needed in RStudio
# pak::pak("reticulate")
reticulate::py_install("PyMuPDF", method = "virtualenv")
reticulate::py_install("Pillow", method = "virtualenv")
reticulate::py_install("opencv-python", method = "virtualenv")
reticulate::py_install("pandas", method = "virtualenv")
reticulate::py_install("matplotlib", method = "virtualenv")
reticulate::py_install("scikit-image", method = "virtualenv")
