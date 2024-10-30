`read-exams ` es un experimento creado en python con ayuda de ChatGPT para leer plantillas de exámenes de alternativas y corregirlas.  


Contiene dos scripts:  

**separate_pdf**: lee un pdf y crea una imágen por página, rotando a orientación portrait si es necesario.

**read_exams**: lee todas las imágenes de la carpeta `input_folder` y las corrige usando las respuestas de la imagen `CORRECT_RESPONSES.png`.

En la carpeta `outputs/[input_folder]` crea:

- un archivo llamado output.csv, con las respuestas de cada imagen y su corrección
- imágenes para cada examen encontrado y procesado correctamente de `input_folder` con un overlay que muestra las respuestas encontradas, y cuales se puntuan como correctas e incorrectas
- imagenes para cada examen encontrado y procesado correctamente con un histograma donde se muestra el punto de corte entre respuestas marcadas y no marcadas. Se usa el método Otsu para establecer el threshold.



Se incluye una carpeta llamada `example` donde está `SCANNED_EXAMS.pdf` y `CORRECT_RESPONSES.png` para que sea sencillo poner a prueba el proceso.  

También se incluye un `template` editable (odg). Si se cambian las posiciones de los recuadros de respuesta, hay que editar los parametros "Response boxes positions" en `read_exams`.