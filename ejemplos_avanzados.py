import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gradio as gr

class EjemplosAvanzados:
    """Ejemplos más avanzados de redes neuronales"""
    
    def __init__(self):
        self.model_mnist = None
        self.history = None
    
    def crear_cnn_mnist(self, epochs=5):
        """Crea y entrena una CNN para clasificar dígitos MNIST"""
        
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
        """Clasifica una imagen subida por el usuario"""
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
        """Ejemplo simple de RNN para análisis de sentimientos"""
        
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

# Crear instancia
ejemplos_avanzados = EjemplosAvanzados()
