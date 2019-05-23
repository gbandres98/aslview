# aslview

Un proyecto para la asignatura de Procesamiento de Imágenes Digitales. Aprendemos a utilizar TensorFlow para implementar un clasificador de imágenes del alfatabeto americano de lenguaje de signos (ASL).

### Requisitos

 - python 3, pip3

### Instalación

 - Instalar las dependecias con pip: `pip install -r requirements.txt`
 - (Opcional) Si se quiere utilizar una GPU Nvidia para entrenar el modelo, instalar CUDA ([https://www.tensorflow.org/install/gpu](tutorial)) y tensorflow-gpu: `pip install tensorflow-gpu`. Probar que la instalación es correcta con `python testCuda.py`.

### Ejecución

Para entrenar el modelo, ejecutar el archivo aslview.py: `python aslview.py`