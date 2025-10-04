"""
Ejemplos Avanzados de Redes Neuronales.

Este módulo contiene implementaciones avanzadas de redes neuronales profundas,
incluyendo Redes Neuronales Convolucionales (CNN) y Redes Neuronales Recurrentes (RNN).

Autor: Efren Bohorquez
Repositorio: https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva
Licencia: MIT
Fecha: Octubre 2025

Clases:
    EjemplosAvanzados: Clase principal que implementa ejemplos de CNN y RNN.

Dependencias:
    - numpy: Operaciones numéricas y matriciales
    - matplotlib: Visualización de gráficos
    - tensorflow/keras: Framework de deep learning
    - gradio: Interfaz de usuario web interactiva
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gradio as gr


class EjemplosAvanzados:
    """
    Clase que implementa ejemplos avanzados de redes neuronales.
    
    Esta clase proporciona implementaciones didácticas de arquitecturas avanzadas
    de redes neuronales, incluyendo CNN para clasificación de imágenes y RNN
    para análisis de secuencias. Está diseñada con fines educativos, priorizando
    la claridad y comprensión sobre la eficiencia computacional.
    
    Attributes:
        model_mnist (keras.Model): Modelo CNN entrenado para clasificar dígitos MNIST.
            Inicialmente None hasta que se entrena con crear_cnn_mnist().
        history (keras.callbacks.History): Historial de entrenamiento del modelo.
            Contiene métricas como precisión y pérdida por época.
    
    Métodos:
        crear_cnn_mnist(epochs): Crea y entrena una CNN para clasificar dígitos.
        clasificar_imagen_upload(imagen): Clasifica una imagen subida por el usuario.
        crear_rnn_texto(): Muestra ejemplo conceptual de RNN para análisis de sentimientos.
    
    Ejemplo:
        >>> ejemplos = EjemplosAvanzados()
        >>> fig, info = ejemplos.crear_cnn_mnist(epochs=5)
        >>> # Ahora el modelo está entrenado y listo para clasificar
        >>> fig_pred, resultado = ejemplos.clasificar_imagen_upload(imagen)
    """
    
    def __init__(self):
        """
        Inicializa la clase EjemplosAvanzados.
        
        Crea una nueva instancia con los atributos del modelo inicializados a None.
        El modelo debe ser entrenado usando crear_cnn_mnist() antes de poder
        usarse para clasificación.
        """
        self.model_mnist = None
        self.history = None
    
    def crear_cnn_mnist(self, epochs=5):
        """
        Crea y entrena una Red Neuronal Convolucional para clasificar dígitos MNIST.
        
        Este método construye una CNN con 3 capas convolucionales, entrena el modelo
        con un subconjunto del dataset MNIST, y genera visualizaciones del proceso
        de entrenamiento y ejemplos de clasificación.
        
        Args:
            epochs (int, optional): Número de épocas de entrenamiento. Por defecto 5.
                Más épocas generalmente mejoran la precisión pero incrementan el tiempo.
        
        Returns:
            tuple: Una tupla conteniendo:
                - fig (matplotlib.figure.Figure): Figura con 4 subgráficas mostrando
                  precisión, pérdida, y ejemplos de dígitos clasificados.
                - info (str): Texto formateado con información detallada sobre la
                  arquitectura del modelo, resultados y explicación didáctica.
        
        Raises:
            Exception: Si hay problemas al cargar el dataset MNIST o durante el
                entrenamiento del modelo.
        
        Notas:
            - Usa solo 1000 muestras de entrenamiento para demostración rápida
            - Evalúa con 200 muestras de test para mantener tiempos razonables
            - Actualiza self.model_mnist y self.history para uso posterior
            - Las imágenes MNIST son de 28x28 píxeles en escala de grises
        
        Arquitectura de la CNN:
            1. Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
            2. Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
            3. Conv2D(64, 3x3) + ReLU
            4. Flatten
            5. Dense(64) + ReLU
            6. Dense(10) + Softmax (clasificación 10 dígitos)
        
        Ejemplo:
            >>> ejemplos = EjemplosAvanzados()
            >>> figura, descripcion = ejemplos.crear_cnn_mnist(epochs=10)
            >>> print(descripcion)  # Muestra arquitectura y resultados
        """
        
        # Cargar datos MNIST
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Preprocesar datos
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convertir etiquetas a categóricas
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # Crear modelo CNN
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Entrenar modelo (usar subset para demo rápida)
        history = model.fit(x_train[:1000], y_train[:1000], 
                           epochs=epochs, 
                           batch_size=32,
                           validation_split=0.2,
                           verbose=0)
        
        # Evaluar modelo
        test_loss, test_acc = model.evaluate(x_test[:200], y_test[:200], verbose=0)
        
        # Crear visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Gráfica de precisión
        axes[0, 0].plot(history.history['accuracy'], label='Entrenamiento')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validación')
        axes[0, 0].set_title('Precisión del Modelo')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Precisión')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfica de pérdida
        axes[0, 1].plot(history.history['loss'], label='Entrenamiento')
        axes[0, 1].plot(history.history['val_loss'], label='Validación')
        axes[0, 1].set_title('Pérdida del Modelo')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Pérdida')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mostrar ejemplos de dígitos
        for i in range(8):
            row = i // 4
            col = i % 4
            if row < 2 and col < 2:
                continue
            ax = axes[1, 0] if i < 4 else axes[1, 1]
            if i == 0:
                axes[1, 0].imshow(x_test[i].reshape(28, 28), cmap='gray')
                axes[1, 0].set_title(f'Ejemplo: {np.argmax(y_test[i])}')
                axes[1, 0].axis('off')
            elif i == 1:
                axes[1, 1].imshow(x_test[i].reshape(28, 28), cmap='gray')
                axes[1, 1].set_title(f'Ejemplo: {np.argmax(y_test[i])}')
                axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        self.model_mnist = model
        self.history = history
        
        return fig, f"""
        **Red Neuronal Convolucional (CNN) para MNIST:**
        
        **Arquitectura:**
        - Capas convolucionales: 3 (32, 64, 64 filtros)
        - Capas de pooling: 2
        - Capas densas: 2 (64 y 10 neuronas)
        - Total de parámetros: {model.count_params():,}
        
        **Resultados:**
        - Precisión en test: {test_acc:.2%}
        - Pérdida en test: {test_loss:.4f}
        - Épocas entrenadas: {epochs}
        
        **Explicación:**
        Las CNN son especialmente buenas para imágenes porque:
        1. Las capas convolucionales detectan características locales
        2. Las capas de pooling reducen la dimensionalidad
        3. Mantienen la estructura espacial de las imágenes
        """
    
    def clasificar_imagen_upload(self, imagen):
        """
        Clasifica una imagen de dígito subida por el usuario usando el modelo CNN entrenado.
        
        Este método toma una imagen proporcionada por el usuario, la preprocesa para
        que coincida con el formato esperado por el modelo MNIST, realiza la predicción
        y genera visualizaciones de los resultados.
        
        Args:
            imagen (PIL.Image.Image or None): Imagen subida por el usuario. Debe ser
                una imagen que contenga un dígito del 0 al 9. Será convertida a escala
                de grises y redimensionada a 28x28 píxeles automáticamente.
        
        Returns:
            tuple or str: Si todo es correcto, devuelve una tupla con:
                - fig (matplotlib.figure.Figure): Visualización con la imagen procesada
                  y un gráfico de barras mostrando las probabilidades de cada dígito.
                - info (str): Texto con la predicción, nivel de confianza y explicación.
                
                Si hay error, devuelve un string con el mensaje de error.
        
        Raises:
            Exception: Captura y devuelve como string cualquier error durante el
                procesamiento o clasificación de la imagen.
        
        Notas:
            - Requiere que el modelo esté entrenado (self.model_mnist no sea None)
            - La imagen se convierte automáticamente a escala de grises
            - Se normaliza dividiendo entre 255.0 para valores en rango [0, 1]
            - Funciona mejor con imágenes de dígitos escritos con trazo grueso
        
        Preprocesamiento:
            1. Conversión a escala de grises (modo 'L')
            2. Redimensionamiento a 28x28 píxeles
            3. Normalización de valores [0-255] a [0-1]
            4. Reshape a (1, 28, 28, 1) para el modelo
        
        Ejemplo:
            >>> from PIL import Image
            >>> imagen = Image.open('digito.png')
            >>> fig, resultado = ejemplos.clasificar_imagen_upload(imagen)
            >>> print(resultado)  # "Predicción: 7, Confianza: 95.3%"
        """
        if imagen is None or self.model_mnist is None:
            return "Por favor, entrena el modelo primero y sube una imagen."
        
        try:
            # Preprocesar imagen
            img = imagen.convert('L')  # Convertir a escala de grises
            img = img.resize((28, 28))  # Redimensionar a 28x28
            img_array = np.array(img) / 255.0  # Normalizar
            img_array = img_array.reshape(1, 28, 28, 1)  # Reshape para el modelo
            
            # Hacer predicción
            prediccion = self.model_mnist.predict(img_array, verbose=0)
            digito_predicho = np.argmax(prediccion)
            confianza = np.max(prediccion)
            
            # Crear visualización
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Mostrar imagen original
            ax1.imshow(img, cmap='gray')
            ax1.set_title(f'Imagen Original\nPredicción: {digito_predicho}')
            ax1.axis('off')
            
            # Mostrar probabilidades
            ax2.bar(range(10), prediccion[0])
            ax2.set_xlabel('Dígito')
            ax2.set_ylabel('Probabilidad')
            ax2.set_title('Distribución de Probabilidades')
            ax2.set_xticks(range(10))
            
            plt.tight_layout()
            
            return fig, f"""
            **Predicción:** {digito_predicho}
            **Confianza:** {confianza:.2%}
            
            La red neuronal está {confianza:.1%} segura de que el dígito es un {digito_predicho}.
            """
            
        except Exception as e:
            return f"Error al procesar la imagen: {str(e)}"
    
    def crear_rnn_texto(self):
        """
        Crea una visualización conceptual de RNN para análisis de sentimientos.
        
        Este método genera una demostración didáctica de cómo funcionan las Redes
        Neuronales Recurrentes (RNN) en el contexto de análisis de sentimientos en texto.
        NO entrena un modelo real, sino que proporciona visualizaciones educativas
        sobre la arquitectura y funcionamiento de las RNN.
        
        Returns:
            tuple: Una tupla conteniendo:
                - fig (matplotlib.figure.Figure): Figura con 4 subgráficas mostrando:
                    1. Arquitectura de la RNN procesando una secuencia de palabras
                    2. Distribución de sentimientos en datos de ejemplo
                    3. Simulación de curva de entrenamiento
                    4. Ejemplos de predicciones con niveles de confianza
                - info (str): Texto explicativo sobre características, aplicaciones
                  y ventajas de las RNN en procesamiento de lenguaje natural.
        
        Características Visualizadas:
            - Flujo de datos a través de células RNN
            - Conexiones recurrentes (memoria temporal)
            - Proceso de clasificación de sentimientos
            - Progreso simulado de entrenamiento
        
        Conceptos Educativos:
            - Procesamiento secuencial de texto
            - Memoria de estados anteriores
            - Clasificación positivo/negativo
            - Aplicaciones en NLP (Natural Language Processing)
        
        Notas:
            - Este método es puramente educativo y no entrena un modelo real
            - Los datos de ejemplo son ficticios para demostración
            - La curva de entrenamiento es simulada matemáticamente
            - Útil para entender conceptos antes de implementar RNN reales
        
        Aplicaciones Mencionadas:
            - Análisis de sentimientos en redes sociales
            - Traducción automática de idiomas
            - Reconocimiento de voz y audio
            - Predicción de series temporales
            - Generación de texto
        
        Ejemplo:
            >>> ejemplos = EjemplosAvanzados()
            >>> fig_rnn, explicacion = ejemplos.crear_rnn_texto()
            >>> print(explicacion)  # Muestra características y aplicaciones
        """
        
        # Datos de ejemplo simple
        textos = [
            "Me encanta este producto",
            "Excelente calidad",
            "Muy bueno",
            "Fantástico servicio",
            "Horrible experiencia",
            "Muy malo",
            "Terrible calidad",
            "No lo recomiendo"
        ]
        
        sentimientos = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=positivo, 0=negativo
        
        # Crear visualización conceptual
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Arquitectura RNN
        ax1 = axes[0, 0]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        
        # Secuencia temporal
        palabras = ["Me", "encanta", "este", "producto"]
        for i, palabra in enumerate(palabras):
            x = 2 + i * 2
            ax1.scatter([x], [6], s=200, c='lightblue')
            ax1.text(x, 5.2, palabra, ha='center', fontsize=10)
            
            # RNN cell
            ax1.scatter([x], [3], s=300, c='orange')
            ax1.text(x, 2.2, f'RNN\n{i+1}', ha='center', fontsize=8)
            
            # Conexiones
            ax1.arrow(x, 5.8, 0, -2.5, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
            if i > 0:
                ax1.arrow(x-2, 3, 1.8, 0, head_width=0.1, head_length=0.2, fc='red', ec='red')
        
        # Salida final
        ax1.scatter([8], [1], s=200, c='lightgreen')
        ax1.text(8, 0.2, 'Sentimiento\nPositivo', ha='center', fontsize=10)
        ax1.arrow(8, 2.8, 0, -1.5, head_width=0.1, head_length=0.2, fc='green', ec='green')
        
        ax1.set_title('Arquitectura RNN para Análisis de Sentimientos')
        ax1.axis('off')
        
        # Distribución de sentimientos
        ax2 = axes[0, 1]
        sentimientos_count = [sum(sentimientos), len(sentimientos) - sum(sentimientos)]
        ax2.pie(sentimientos_count, labels=['Positivo', 'Negativo'], autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'])
        ax2.set_title('Distribución de Sentimientos en Datos')
        
        # Simulación de entrenamiento
        ax3 = axes[1, 0]
        epochs = np.arange(1, 21)
        accuracy = 0.5 + 0.4 * (1 - np.exp(-epochs/5)) + 0.05 * np.random.randn(20)
        ax3.plot(epochs, accuracy, 'b-', linewidth=2)
        ax3.set_xlabel('Épocas')
        ax3.set_ylabel('Precisión')
        ax3.set_title('Progreso del Entrenamiento')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0.4, 1.0)
        
        # Ejemplo de predicción
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, 'Ejemplos de Predicción:', fontsize=12, weight='bold', transform=ax4.transAxes)
        
        ejemplos = [
            ("'Me encanta'", "Positivo", 0.85),
            ("'Muy malo'", "Negativo", 0.92),
            ("'Está bien'", "Neutral", 0.60)
        ]
        
        for i, (texto, pred, conf) in enumerate(ejemplos):
            y_pos = 0.6 - i * 0.15
            ax4.text(0.1, y_pos, f"{texto} → {pred} ({conf:.0%})", 
                    transform=ax4.transAxes, fontsize=10)
        
        ax4.set_title('Ejemplos de Clasificación')
        ax4.axis('off')
        
        plt.tight_layout()
        
        return fig, """
        **Red Neuronal Recurrente (RNN) para Análisis de Sentimientos:**
        
        **Características de las RNN:**
        - Procesan secuencias de datos (texto, tiempo, etc.)
        - Tienen "memoria" de entradas anteriores
        - Útiles para tareas como traducción, análisis de sentimientos
        
        **Aplicaciones:**
        - Análisis de sentimientos en redes sociales
        - Traducción automática
        - Reconocimiento de voz
        - Predicción de series temporales
        
        **Ventajas:**
        - Manejan secuencias de longitud variable
        - Capturan dependencias temporales
        - Flexibles para diferentes tipos de entrada
        """

# Crear instancia global para uso en la interfaz Gradio
ejemplos_avanzados = EjemplosAvanzados()
