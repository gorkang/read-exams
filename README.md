`read-exams ` es un experimento creado en python con ayuda de ChatGPT para leer plantillas de exámenes de alternativas y corregirlas.  

## Como usar

### Archivos

1. Crea una carpeta en `inputs` con el nombre de tu asignatura-prueba. Por ejemplo, `inputs/example`
2. Copia tus exámenes escaneados dentro de esa carpeta: `SCANNED_EXAMS.pdf`


### Scripts

Una vez tenemos la plantilla y los exámenes en pdf en la carpeta en `inputs/example` :

1. Usar `separate_pdf.py`, cambiando el parámetro `input_folder`.  
2. Renombrar la plantilla con las respuestas correctas a `CORRECT_RESPONSES.png`.  
3. Haz una copia de `parameters_example.py`, cambiando los parámetros `input_subfolder` y `num_questions`, etc.

En la carpeta `output/[input_subfolder]` se generará un archivo `output.csv` con las correcciones, además de carpetas `output_corners`, `output_intensities` y `output_images` con los detalles para cada uno de los exámenes.


## Dependencias

### Ubuntu 24.04
sudo apt install python3-ipykernel python3-opencv python3-pandas python3-skimage python3-fitz



Contiene dos scripts:  

**separate_pdf**: lee un pdf y crea una imágen por página, rotando a orientación portrait si es necesario.

**parameters_example.py**: lee todas las imágenes de la carpeta `input_subfolder` y las corrige usando las respuestas de la imagen `CORRECT_RESPONSES.png`.

En la carpeta `outputs/[input_subfolder]` crea:

- un archivo llamado output.csv, con las respuestas de cada imagen y su corrección
- imágenes para cada examen encontrado y procesado correctamente de `input_subfolder` con un overlay que muestra las respuestas encontradas, y cuales se puntuan como correctas e incorrectas
- imagenes para cada examen encontrado y procesado correctamente con un histograma donde se muestra el punto de corte entre respuestas marcadas y no marcadas. Se usa el método Otsu para establecer el umbral.



Se incluye una carpeta llamada `example` donde está `SCANNED_EXAMS.pdf` y `CORRECT_RESPONSES.png` para que sea sencillo poner a prueba el proceso.  

También se incluye un `template` editable (odg). Si se cambian las posiciones de los recuadros de respuesta, hay que editar los parametros "Response boxes positions" en `py/parameters_global.py`.