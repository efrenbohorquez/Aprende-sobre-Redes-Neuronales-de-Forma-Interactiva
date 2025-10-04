"""
Aplicación Educativa de Redes Neuronales con Gradio
=====================================================

Este módulo implementa una aplicación interactiva para enseñar conceptos
fundamentales de redes neuronales utilizando Gradio como interfaz web.

Autor: Efren Bohorquez
Proyecto: Aprende sobre Redes Neuronales de Forma Interactiva
Repositorio: https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva
Licencia: MIT

Características principales:
    - Simulación de neurona artificial
    - Visualización de funciones de activación
    - Entrenamiento de redes neuronales multicapa
    - Demostración de backpropagation
    - CNN para reconocimiento de dígitos (MNIST)
    - RNN para análisis de sentimientos

Dependencias:
    - gradio >= 4.0.0
    - numpy >= 1.21.0
    - matplotlib >= 3.5.0
    - tensorflow >= 2.13.0
    - scikit-learn >= 1.3.0
    - pandas >= 2.0.0
    - seaborn >= 0.12.0
    - plotly >= 5.17.0
    - pillow >= 10.0.0
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
# TensorFlow se cargará solo cuando sea necesario (lazy loading)
from sklearn.datasets import make_classification, make_circles
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns
from PIL import Image
import io
import base64

# No importar ejemplos avanzados al inicio para evitar cargar TensorFlow
# Se cargará cuando el usuario acceda a esas pestañas

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.style.use('default')


class RedNeuronalEducativa:
    """
    Clase principal para la aplicación educativa de redes neuronales.
    
    Esta clase proporciona métodos interactivos para demostrar conceptos
    fundamentales de redes neuronales, incluyendo neuronas individuales,
    funciones de activación, arquitecturas de red y algoritmos de aprendizaje.
    
    Attributes:
        model: Modelo de red neuronal (inicialmente None)
        ejemplos_avanzados: Instancia de ejemplos avanzados (carga bajo demanda)
    
    Methods:
        crear_neurona_simple: Simula el comportamiento de una neurona artificial
        visualizar_funciones_activacion: Compara diferentes funciones de activación
        crear_red_simple: Entrena una red neuronal multicapa
        explicar_backpropagation: Visualiza el proceso de aprendizaje
        entrenar_cnn: Carga y entrena una CNN para MNIST
        clasificar_imagen: Clasifica imágenes de dígitos
        mostrar_rnn: Muestra la arquitectura de una RNN
    """
    
    def __init__(self):
        """Inicializa la clase sin cargar TensorFlow."""
        self.model = None
        self.ejemplos_avanzados = None  # Se cargará bajo demanda
    
    def _cargar_ejemplos_avanzados(self):
        """
        Carga los ejemplos avanzados solo cuando sean necesarios.
        
        Implementa lazy loading para evitar cargar TensorFlow al inicio,
        lo que reduce significativamente el tiempo de arranque de la aplicación.
        
        Returns:
            EjemplosAvanzados: Instancia de la clase de ejemplos avanzados
        """
        if self.ejemplos_avanzados is None:
            from ejemplos_avanzados import EjemplosAvanzados
            self.ejemplos_avanzados = EjemplosAvanzados()
        return self.ejemplos_avanzados
        
    def crear_neurona_simple(self, entrada1, entrada2, peso1, peso2, sesgo, funcion_activacion):
        """
        Simula el comportamiento de una neurona artificial.
        
        Esta función demuestra cómo una neurona procesa entradas mediante
        pesos y un sesgo, y luego aplica una función de activación.
        
        Args:
            entrada1 (float): Primera entrada (x1)
            entrada2 (float): Segunda entrada (x2)
            peso1 (float): Peso de la primera entrada (w1)
            peso2 (float): Peso de la segunda entrada (w2)
            sesgo (float): Sesgo o bias de la neurona (b)
            funcion_activacion (str): Función de activación a aplicar
                                     ("Sigmoid", "ReLU", "Tanh", "Linear")
        
        Returns:
            matplotlib.figure.Figure: Gráfico con visualización de la neurona
                                     y el proceso de cálculo
        """
        # Calcular la suma ponderada
        suma_ponderada = (entrada1 * peso1) + (entrada2 * peso2) + sesgo
        
        # Aplicar función de activación
        if funcion_activacion == "Sigmoid":
            salida = 1 / (1 + np.exp(-suma_ponderada))
        elif funcion_activacion == "ReLU":
            salida = max(0, suma_ponderada)
        elif funcion_activacion == "Tanh":
            salida = np.tanh(suma_ponderada)
        else:  # Linear
            salida = suma_ponderada
            
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Diagrama de la neurona
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 6)
        
        # Entradas
        ax1.scatter([1], [5], s=200, c='lightblue', label=f'Entrada 1: {entrada1}')
        ax1.scatter([1], [1], s=200, c='lightblue', label=f'Entrada 2: {entrada2}')
        
        # Neurona
        ax1.scatter([6], [3], s=500, c='orange', label='Neurona')
        
        # Salida
        ax1.scatter([9], [3], s=200, c='lightgreen', label=f'Salida: {salida:.3f}')
        
        # Conexiones con pesos
        ax1.arrow(1.5, 5, 4, -1.5, head_width=0.1, head_length=0.2, fc='red', ec='red')
        ax1.text(3, 4.5, f'w1={peso1}', fontsize=10, color='red')
        
        ax1.arrow(1.5, 1, 4, 1.5, head_width=0.1, head_length=0.2, fc='red', ec='red')
        ax1.text(3, 1.5, f'w2={peso2}', fontsize=10, color='red')
        
        ax1.arrow(6.5, 3, 2, 0, head_width=0.1, head_length=0.2, fc='green', ec='green')
        
        # Sesgo
        ax1.text(6, 2, f'Sesgo: {sesgo}', fontsize=10, color='purple')
        ax1.text(6, 4, f'Función: {funcion_activacion}', fontsize=10, color='blue')
        
        ax1.set_title('Neurona Artificial')
        ax1.legend()
        ax1.axis('off')
        
        # Gráfica de función de activación
        x = np.linspace(-5, 5, 100)
        if funcion_activacion == "Sigmoid":
            y = 1 / (1 + np.exp(-x))
        elif funcion_activacion == "ReLU":
            y = np.maximum(0, x)
        elif funcion_activacion == "Tanh":
            y = np.tanh(x)
        else:  # Linear
            y = x
            
        ax2.plot(x, y, 'b-', linewidth=2)
        ax2.scatter([suma_ponderada], [salida], s=100, c='red', zorder=5)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Entrada (suma ponderada)')
        ax2.set_ylabel('Salida')
        ax2.set_title(f'Función de Activación: {funcion_activacion}')
        
        plt.tight_layout()
        return fig, f"""
        **Resultado:**
        
        Suma ponderada = ({entrada1} × {peso1}) + ({entrada2} × {peso2}) + {sesgo} = {suma_ponderada:.3f}
        
        Salida después de {funcion_activacion} = {salida:.3f}
        
        **Explicación:**
        Una neurona artificial toma múltiples entradas, las multiplica por sus respectivos pesos,
        suma todo incluyendo un sesgo, y aplica una función de activación para producir la salida.
        """
    
    def visualizar_funciones_activacion(self):
        """Muestra las diferentes funciones de activación"""
        x = np.linspace(-5, 5, 100)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        funciones = {
            'Sigmoid': 1 / (1 + np.exp(-x)),
            'ReLU': np.maximum(0, x),
            'Tanh': np.tanh(x),
            'Leaky ReLU': np.where(x > 0, x, 0.01 * x)
        }
        
        for i, (nombre, y) in enumerate(funciones.items()):
            axes[i].plot(x, y, 'b-', linewidth=2)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'{nombre}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('f(x)')
            
        plt.tight_layout()
        return fig
    
    def crear_red_simple(self, capas_ocultas, neuronas_por_capa, datos_tipo):
        """Crea y entrena una red neuronal simple"""
        # Generar datos
        if datos_tipo == "Clasificación lineal":
            X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                     n_informative=2, n_clusters_per_class=1, random_state=42)
        else:  # Círculos concéntricos
            X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
        
        # Crear el modelo
        capas = [neuronas_por_capa] * capas_ocultas
        modelo = MLPClassifier(hidden_layer_sizes=capas, max_iter=1000, random_state=42)
        modelo.fit(X, y)
        
        # Crear malla para visualización
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predecir en la malla
        Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Datos originales
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax1.set_title(f'Datos de Entrenamiento: {datos_tipo}')
        ax1.set_xlabel('Característica 1')
        ax1.set_ylabel('Característica 2')
        plt.colorbar(scatter, ax=ax1)
        
        # Límites de decisión
        ax2.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax2.set_title(f'Límites de Decisión\nRed: {capas_ocultas} capas × {neuronas_por_capa} neuronas')
        ax2.set_xlabel('Característica 1')
        ax2.set_ylabel('Característica 2')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        
        # Calcular precisión
        precision = modelo.score(X, y)
        
        return fig, f"""
        **Configuración de la Red:**
        - Capas ocultas: {capas_ocultas}
        - Neuronas por capa: {neuronas_por_capa}
        - Total de parámetros: ~{sum(capas) * 2 + sum(capas) + capas_ocultas}
        
        **Resultado:**
        - Precisión en entrenamiento: {precision:.2%}
        
        **Explicación:**
        Esta red neuronal aprende a separar las dos clases de datos creando límites de decisión no lineales.
        Más capas y neuronas permiten límites más complejos, pero pueden causar sobreajuste.
        """
    
    def explicar_backpropagation(self):
        """Visualiza el proceso de backpropagation"""
        # Crear una red simple para demostrar
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Forward pass
        ax1 = axes[0, 0]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        
        # Neuronas
        ax1.scatter([2], [6], s=300, c='lightblue', label='Entrada')
        ax1.scatter([5], [7], s=300, c='orange', label='Oculta 1')
        ax1.scatter([5], [5], s=300, c='orange', label='Oculta 2')
        ax1.scatter([8], [6], s=300, c='lightgreen', label='Salida')
        
        # Conexiones forward
        for start_x, start_y in [(2, 6)]:
            for end_x, end_y in [(5, 7), (5, 5)]:
                ax1.arrow(start_x+0.2, start_y, end_x-start_x-0.4, end_y-start_y, 
                         head_width=0.1, head_length=0.2, fc='blue', ec='blue', alpha=0.7)
        
        for start_x, start_y in [(5, 7), (5, 5)]:
            ax1.arrow(start_x+0.2, start_y, 8-start_x-0.4, 6-start_y, 
                     head_width=0.1, head_length=0.2, fc='blue', ec='blue', alpha=0.7)
        
        ax1.set_title('Forward Pass')
        ax1.text(5, 2, 'Los datos fluyen hacia adelante', ha='center', fontsize=12)
        ax1.legend()
        ax1.axis('off')
        
        # Backward pass
        ax2 = axes[0, 1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        
        # Neuronas
        ax2.scatter([2], [6], s=300, c='lightblue', label='Entrada')
        ax2.scatter([5], [7], s=300, c='orange', label='Oculta 1')
        ax2.scatter([5], [5], s=300, c='orange', label='Oculta 2')
        ax2.scatter([8], [6], s=300, c='lightgreen', label='Salida')
        
        # Conexiones backward
        for end_x, end_y in [(5, 7), (5, 5)]:
            ax2.arrow(8-0.2, 6, end_x-8+0.4, end_y-6, 
                     head_width=0.1, head_length=0.2, fc='red', ec='red', alpha=0.7)
        
        for end_x, end_y in [(2, 6)]:
            for start_x, start_y in [(5, 7), (5, 5)]:
                ax2.arrow(start_x-0.2, start_y, end_x-start_x+0.4, end_y-start_y, 
                         head_width=0.1, head_length=0.2, fc='red', ec='red', alpha=0.7)
        
        ax2.set_title('Backward Pass (Backpropagation)')
        ax2.text(5, 2, 'Los errores se propagan hacia atrás', ha='center', fontsize=12)
        ax2.legend()
        ax2.axis('off')
        
        # Gráfica de función de pérdida
        ax3 = axes[1, 0]
        epochs = np.arange(1, 101)
        loss = 2 * np.exp(-epochs/20) + 0.1 + 0.05 * np.random.randn(100)
        ax3.plot(epochs, loss, 'b-', linewidth=2)
        ax3.set_xlabel('Épocas')
        ax3.set_ylabel('Función de Pérdida')
        ax3.set_title('Reducción del Error Durante el Entrenamiento')
        ax3.grid(True, alpha=0.3)
        
        # Actualización de pesos
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, 'Proceso de Backpropagation:', fontsize=14, weight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, '1. Forward Pass: Calcular predicción', transform=ax4.transAxes)
        ax4.text(0.1, 0.6, '2. Calcular error (pérdida)', transform=ax4.transAxes)
        ax4.text(0.1, 0.5, '3. Backward Pass: Calcular gradientes', transform=ax4.transAxes)
        ax4.text(0.1, 0.4, '4. Actualizar pesos usando gradientes', transform=ax4.transAxes)
        ax4.text(0.1, 0.3, '5. Repetir hasta convergencia', transform=ax4.transAxes)
        ax4.text(0.1, 0.1, 'Fórmula: peso_nuevo = peso_viejo - α × gradiente', 
                transform=ax4.transAxes, style='italic')
        ax4.set_title('Algoritmo de Aprendizaje')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def demostrar_callbacks_optimizacion(self, tecnica_seleccionada, epochs=50):
        """
        Demuestra técnicas de optimización y callbacks para entrenamiento de redes neuronales.
        
        Este método visualiza diferentes estrategias de optimización como Early Stopping,
        Learning Rate Scheduling, y Model Checkpointing para mejorar el entrenamiento.
        
        Args:
            tecnica_seleccionada (str): Técnica a demostrar ("Early Stopping", "Learning Rate Decay", 
                                       "Reducir LR en Plateau", "Todas las Técnicas")
            epochs (int): Número de épocas para la simulación
        
        Returns:
            tuple: Figura con visualizaciones y texto explicativo
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Callbacks y Optimización: {tecnica_seleccionada}', fontsize=16, weight='bold')
        
        epochs_range = np.arange(1, epochs + 1)
        
        # Simular datos de entrenamiento básicos
        np.random.seed(42)
        base_train_loss = 2.5 * np.exp(-epochs_range/15) + 0.1 + 0.05 * np.random.randn(epochs)
        base_val_loss = 2.8 * np.exp(-epochs_range/18) + 0.15 + 0.08 * np.random.randn(epochs)
        
        # === Gráfica 1: Early Stopping ===
        ax1 = axes[0, 0]
        
        if tecnica_seleccionada in ["Early Stopping", "Todas las Técnicas"]:
            # Simular early stopping
            best_epoch = 25
            train_loss_es = base_train_loss.copy()
            val_loss_es = base_val_loss.copy()
            # Simular overfitting después del mejor punto
            val_loss_es[best_epoch:] += np.linspace(0, 0.3, epochs - best_epoch)
            
            ax1.plot(epochs_range, train_loss_es, 'b-', label='Pérdida Entrenamiento', linewidth=2)
            ax1.plot(epochs_range, val_loss_es, 'r-', label='Pérdida Validación', linewidth=2)
            ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Early Stop (época {best_epoch})')
            ax1.scatter([best_epoch], [val_loss_es[best_epoch-1]], color='green', s=200, zorder=5, marker='*')
            
            info_es = f"""
**Early Stopping Activado:**
- Mejor época: {best_epoch}
- Paciencia: 5 épocas
- Evita overfitting
- Ahorra tiempo de entrenamiento
"""
        else:
            ax1.plot(epochs_range, base_train_loss, 'b-', label='Entrenamiento', linewidth=2)
            ax1.plot(epochs_range, base_val_loss, 'r-', label='Validación', linewidth=2)
            info_es = "Early Stopping no aplicado"
        
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Pérdida')
        ax1.set_title('Early Stopping: Detener cuando deja de mejorar')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === Gráfica 2: Learning Rate Decay ===
        ax2 = axes[0, 1]
        
        if tecnica_seleccionada in ["Learning Rate Decay", "Todas las Técnicas"]:
            # Simular decay exponencial
            initial_lr = 0.1
            decay_rate = 0.95
            lr_exponential = initial_lr * (decay_rate ** epochs_range)
            
            # Simular decay escalonado
            lr_step = np.ones(epochs) * initial_lr
            lr_step[10:20] = initial_lr * 0.5
            lr_step[20:30] = initial_lr * 0.25
            lr_step[30:40] = initial_lr * 0.125
            lr_step[40:] = initial_lr * 0.0625
            
            ax2.plot(epochs_range, lr_exponential, 'g-', label='Decay Exponencial', linewidth=2)
            ax2.plot(epochs_range, lr_step, 'orange', label='Decay Escalonado', linewidth=2, linestyle='--')
            
            info_lr = f"""
**Learning Rate Decay:**
- LR inicial: {initial_lr}
- Tasa de decay: {decay_rate}
- Permite ajustes finos al final
- Mejora convergencia
"""
        else:
            constant_lr = np.ones(epochs) * 0.01
            ax2.plot(epochs_range, constant_lr, 'b-', label='LR Constante', linewidth=2)
            info_lr = "Learning Rate constante (sin decay)"
        
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Decay: Reducción gradual del LR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # === Gráfica 3: ReduceLROnPlateau ===
        ax3 = axes[1, 0]
        
        if tecnica_seleccionada in ["Reducir LR en Plateau", "Todas las Técnicas"]:
            # Simular pérdida con plateaus
            loss_plateau = np.concatenate([
                2.0 - np.linspace(0, 1.0, 15),  # Mejora inicial
                1.0 + 0.05 * np.random.randn(10),  # Plateau 1
                1.0 - np.linspace(0, 0.4, 10),  # Mejora después de reducir LR
                0.6 + 0.03 * np.random.randn(10),  # Plateau 2
                0.6 - np.linspace(0, 0.2, epochs - 45)  # Mejora final
            ])
            
            # Marcar puntos donde se reduce LR
            reduce_points = [25, 35]
            
            ax3.plot(epochs_range, loss_plateau, 'purple', linewidth=2, label='Pérdida de Validación')
            for point in reduce_points:
                ax3.axvline(x=point, color='red', linestyle='--', alpha=0.7, label=f'LR reducido' if point == reduce_points[0] else '')
                ax3.scatter([point], [loss_plateau[point-1]], color='red', s=150, zorder=5)
            
            info_plateau = f"""
**ReduceLR On Plateau:**
- Factor de reducción: 0.5
- Paciencia: 5 épocas
- Reduce LR cuando no mejora
- Adaptativo al entrenamiento
"""
        else:
            ax3.plot(epochs_range, base_val_loss, 'b-', linewidth=2)
            info_plateau = "ReduceLR no aplicado"
        
        ax3.set_xlabel('Épocas')
        ax3.set_ylabel('Pérdida de Validación')
        ax3.set_title('ReduceLR On Plateau: LR adaptativo según progreso')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # === Gráfica 4: Comparación de Técnicas ===
        ax4 = axes[1, 1]
        
        # Simular resultados finales con diferentes técnicas
        tecnicas = ['Sin\nOptimización', 'Early\nStopping', 'LR Decay', 'ReduceLR\nPlateau', 'Todas\nCombinadas']
        val_accuracy = [0.75, 0.82, 0.84, 0.83, 0.89]
        train_time = [100, 60, 100, 95, 65]  # Tiempo relativo
        
        x_pos = np.arange(len(tecnicas))
        bars = ax4.bar(x_pos, val_accuracy, color=['gray', 'green', 'blue', 'purple', 'gold'], alpha=0.7)
        
        # Añadir etiquetas de tiempo
        for i, (bar, time) in enumerate(zip(bars, train_time)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val_accuracy[i]:.2f}\n({time}%)',
                    ha='center', va='bottom', fontsize=9)
        
        ax4.set_xlabel('Técnica de Optimización')
        ax4.set_ylabel('Precisión en Validación')
        ax4.set_title('Comparación de Técnicas (Precisión y Tiempo)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(tecnicas, rotation=0, ha='center')
        ax4.set_ylim(0.7, 0.95)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Generar explicación detallada
        explicacion = f"""
## 🎯 Callbacks y Técnicas de Optimización

### Técnica Seleccionada: **{tecnica_seleccionada}**

---

### 📊 **1. Early Stopping**
{info_es}

**¿Cuándo usar?**
- Cuando hay riesgo de overfitting
- Para ahorrar tiempo de entrenamiento
- En datasets pequeños

**Código ejemplo:**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

---

### 📉 **2. Learning Rate Decay**
{info_lr}

**¿Cuándo usar?**
- En entrenamientos largos
- Para ajuste fino al final
- Cuando la pérdida oscila

**Código ejemplo:**
```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    return lr * 0.95
    
lr_decay = LearningRateScheduler(scheduler)
```

---

### 🔄 **3. ReduceLR On Plateau**
{info_plateau}

**¿Cuándo usar?**
- Cuando no sabes el mejor LR
- En problemas complejos
- Para adaptación automática

**Código ejemplo:**
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

---

### 💾 **4. Model Checkpoint (Bonus)**

Guarda el mejor modelo durante el entrenamiento:

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)
```

---

### 🎓 **Mejores Prácticas:**

1. **Combina callbacks**: Early Stopping + ReduceLR funciona muy bien
2. **Monitorea val_loss**: No solo la precisión
3. **Ajusta la paciencia**: Más paciencia para datasets grandes
4. **Guarda checkpoints**: Siempre guarda el mejor modelo
5. **Experimenta**: Prueba diferentes combinaciones

---

### 📈 **Resultados Esperados:**

- **Sin optimización**: Riesgo de overfitting, tiempo desperdiciado
- **Con callbacks**: Mejor generalización, entrenamiento eficiente
- **Combinadas**: Óptimo rendimiento y tiempo

**¡Las callbacks son esenciales para entrenar modelos profesionales!** 🚀
"""
        
        return fig, explicacion
    
    # Métodos wrapper para ejemplos avanzados (cargan TensorFlow solo cuando se usan)
    def entrenar_cnn(self, epochs):
        """Wrapper para entrenar CNN"""
        ejemplos = self._cargar_ejemplos_avanzados()
        return ejemplos.crear_cnn_mnist(epochs)
    
    def clasificar_imagen(self, imagen):
        """Wrapper para clasificar imagen"""
        ejemplos = self._cargar_ejemplos_avanzados()
        return ejemplos.clasificar_imagen_upload(imagen)
    
    def mostrar_rnn(self):
        """Wrapper para mostrar RNN"""
        ejemplos = self._cargar_ejemplos_avanzados()
        return ejemplos.crear_rnn_texto()

# Crear instancia de la clase
red_educativa = RedNeuronalEducativa()

# Crear la interfaz de Gradio mejorada
with gr.Blocks(title="🧠 Redes Neuronales Interactivas") as demo:
    gr.Markdown("""
    # 🧠 Aprende sobre Redes Neuronales de Forma Interactiva
    
    Esta aplicación te ayudará a entender los conceptos fundamentales de las redes neuronales
    a través de visualizaciones interactivas y ejemplos prácticos avanzados.
    """)
    
    with gr.Tabs():
        # Tab 1: Neurona Simple
        with gr.Tab("🔗 Neurona Artificial"):
            gr.Markdown("## Explora cómo funciona una neurona artificial")
            
            with gr.Row():
                with gr.Column():
                    entrada1 = gr.Slider(-5, 5, value=1, label="Entrada 1", step=0.1)
                    entrada2 = gr.Slider(-5, 5, value=0.5, label="Entrada 2", step=0.1)
                    peso1 = gr.Slider(-2, 2, value=0.7, label="Peso 1", step=0.1)
                    peso2 = gr.Slider(-2, 2, value=0.3, label="Peso 2", step=0.1)
                    sesgo = gr.Slider(-2, 2, value=0, label="Sesgo", step=0.1)
                    funcion = gr.Dropdown(["Sigmoid", "ReLU", "Tanh", "Linear"], 
                                        value="Sigmoid", label="Función de Activación")
                    
                with gr.Column():
                    plot_neurona = gr.Plot()
                    explicacion_neurona = gr.Markdown()
            
            # Conectar controles con función
            controles_neurona = [entrada1, entrada2, peso1, peso2, sesgo, funcion]
            for control in controles_neurona:
                control.change(red_educativa.crear_neurona_simple, 
                             inputs=controles_neurona, 
                             outputs=[plot_neurona, explicacion_neurona])
        
        # Tab 2: Funciones de Activación
        with gr.Tab("📊 Funciones de Activación"):
            gr.Markdown("## Comprende las diferentes funciones de activación")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    **Funciones de Activación:**
                    
                    - **Sigmoid**: Salida entre 0 y 1, útil para probabilidades
                    - **ReLU**: Más rápida, evita el problema del gradiente que desaparece
                    - **Tanh**: Salida entre -1 y 1, centrada en cero
                    - **Leaky ReLU**: Permite gradientes pequeños para valores negativos
                    """)
                
                with gr.Column():
                    plot_activaciones = gr.Plot(red_educativa.visualizar_funciones_activacion())
        
        # Tab 3: Red Neuronal Simple
        with gr.Tab("🕸️ Red Neuronal"):
            gr.Markdown("## Experimenta con diferentes arquitecturas de red")
            
            with gr.Row():
                with gr.Column():
                    capas = gr.Slider(1, 5, value=2, step=1, label="Número de Capas Ocultas")
                    neuronas = gr.Slider(5, 50, value=10, step=5, label="Neuronas por Capa")
                    tipo_datos = gr.Dropdown(["Clasificación lineal", "Círculos concéntricos"], 
                                           value="Clasificación lineal", label="Tipo de Datos")
                    btn_entrenar = gr.Button("🚀 Entrenar Red", variant="primary")
                
                with gr.Column():
                    plot_red = gr.Plot()
                    resultado_red = gr.Markdown()
            
            btn_entrenar.click(red_educativa.crear_red_simple, 
                             inputs=[capas, neuronas, tipo_datos], 
                             outputs=[plot_red, resultado_red])
        
        # Tab 4: Backpropagation
        with gr.Tab("🔄 Backpropagation"):
            gr.Markdown("## Entiende cómo aprenden las redes neuronales")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    **Backpropagation** es el algoritmo que permite a las redes neuronales aprender:
                    
                    1. **Forward Pass**: Los datos fluyen hacia adelante para hacer una predicción
                    2. **Cálculo del Error**: Se compara la predicción con el resultado esperado
                    3. **Backward Pass**: El error se propaga hacia atrás para calcular gradientes
                    4. **Actualización**: Los pesos se ajustan para reducir el error
                    
                    Este proceso se repite muchas veces hasta que la red aprende los patrones.
                    """)
                
                with gr.Column():
                    plot_backprop = gr.Plot(red_educativa.explicar_backpropagation())
        
        # Tab 5: CNN para MNIST
        with gr.Tab("🖼️ CNN - Reconocimiento de Dígitos"):
            gr.Markdown("## Red Neuronal Convolucional para clasificar dígitos escritos a mano")
            
            with gr.Row():
                with gr.Column():
                    epochs_cnn = gr.Slider(1, 10, value=3, step=1, label="Épocas de Entrenamiento")
                    btn_entrenar_cnn = gr.Button("🚀 Entrenar CNN", variant="primary")
                    
                    gr.Markdown("### Prueba tu propia imagen:")
                    imagen_upload = gr.Image(type="pil", label="Sube una imagen de un dígito (0-9)")
                    btn_clasificar = gr.Button("🔍 Clasificar Imagen")
                
                with gr.Column():
                    plot_cnn = gr.Plot()
                    resultado_cnn = gr.Markdown()
                    
                    # Resultados de clasificación
                    plot_clasificacion = gr.Plot()
                    resultado_clasificacion = gr.Markdown()
            
            btn_entrenar_cnn.click(red_educativa.entrenar_cnn, 
                                 inputs=[epochs_cnn], 
                                 outputs=[plot_cnn, resultado_cnn])
            
            btn_clasificar.click(red_educativa.clasificar_imagen,
                               inputs=[imagen_upload],
                               outputs=[plot_clasificacion, resultado_clasificacion])
        
        # Tab 6: RNN para Análisis de Sentimientos
        with gr.Tab("📝 RNN - Análisis de Sentimientos"):
            gr.Markdown("## Red Neuronal Recurrente para procesar secuencias de texto")
            
            with gr.Row():
                with gr.Column():
                    btn_mostrar_rnn = gr.Button("📊 Mostrar Arquitectura RNN", variant="primary")
                    
                    gr.Markdown("""
                    ### Características de las RNN:
                    - Procesan datos secuenciales (texto, audio, series temporales)
                    - Tienen "memoria" de entradas anteriores
                    - Ideales para tareas de procesamiento de lenguaje natural
                    """)
                
                with gr.Column():
                    plot_rnn = gr.Plot()
                    resultado_rnn = gr.Markdown()
            
            btn_mostrar_rnn.click(red_educativa.mostrar_rnn,
                                outputs=[plot_rnn, resultado_rnn])
        
        # Tab 7: Callbacks y Optimización
        with gr.Tab("⚙️ Callbacks y Optimización"):
            gr.Markdown("## Técnicas avanzadas para mejorar el entrenamiento de redes neuronales")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### ¿Qué son los Callbacks?
                    
                    Los **callbacks** son funciones que se ejecutan durante el entrenamiento para:
                    - **Monitorear** el progreso
                    - **Ajustar** hiperparámetros automáticamente
                    - **Detener** el entrenamiento cuando es apropiado
                    - **Guardar** el mejor modelo
                    
                    Son esenciales para entrenar modelos de forma eficiente y profesional.
                    """)
                    
                    tecnica_callback = gr.Radio(
                        choices=[
                            "Early Stopping",
                            "Learning Rate Decay", 
                            "Reducir LR en Plateau",
                            "Todas las Técnicas"
                        ],
                        value="Todas las Técnicas",
                        label="Selecciona la técnica a explorar"
                    )
                    
                    epochs_callback = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Épocas de simulación"
                    )
                    
                    btn_demostrar_callbacks = gr.Button(
                        "🚀 Demostrar Técnicas de Optimización",
                        variant="primary"
                    )
                
                with gr.Column():
                    plot_callbacks = gr.Plot()
                    resultado_callbacks = gr.Markdown()
            
            btn_demostrar_callbacks.click(
                red_educativa.demostrar_callbacks_optimizacion,
                inputs=[tecnica_callback, epochs_callback],
                outputs=[plot_callbacks, resultado_callbacks]
            )
        
        # Tab 8: Recursos Adicionales
        with gr.Tab("📚 Recursos"):
            gr.Markdown("""
            ## 📖 Aprende Más sobre Redes Neuronales
            
            ### 🎯 Conceptos Clave que has Aprendido:
            
            1. **Neurona Artificial**: Unidad básica que procesa información
            2. **Pesos y Sesgos**: Parámetros que la red ajusta durante el aprendizaje
            3. **Funciones de Activación**: Introducen no-linealidad en el modelo
            4. **Arquitectura**: El número y tamaño de las capas afecta la capacidad
            5. **Backpropagation**: Algoritmo que permite el aprendizaje automático
            6. **CNN**: Especializadas en procesamiento de imágenes
            7. **RNN**: Ideales para datos secuenciales como texto
            8. **Callbacks**: Técnicas para optimizar y controlar el entrenamiento
            
            ### 🚀 Próximos Pasos:
            
            - Experimenta con diferentes configuraciones
            - Prueba con tus propios datos
            - Explora redes más complejas (LSTM, Transformers)
            - Aprende sobre regularización y optimización
            - Investiga aplicaciones específicas (NLP, Computer Vision)
            
            ### 🛠️ Herramientas Utilizadas:
            
            - **Gradio**: Para la interfaz interactiva
            - **TensorFlow**: Para modelos de deep learning
            - **Scikit-learn**: Para modelos básicos de ML
            - **Matplotlib/Plotly**: Para las visualizaciones
            - **NumPy**: Para cálculos numéricos
            
            ### 📋 Ejercicios Propuestos:
            
            1. **Experimenta con la neurona**: Cambia los pesos y observa cómo afecta la salida
            2. **Compara funciones de activación**: ¿Cuál es mejor para diferentes problemas?
            3. **Arquitecturas de red**: ¿Más capas siempre es mejor?
            4. **Datos complejos**: Prueba la red con círculos concéntricos
            5. **CNN personalizada**: Entrena con diferentes números de épocas
            6. **Análisis de errores**: ¿Qué tipos de dígitos confunde más la CNN?
            7. **Optimización**: Compara diferentes combinaciones de callbacks
            8. **Learning Rate**: ¿Cómo afecta el LR al tiempo de convergencia?
            """)

if __name__ == "__main__":
    demo.launch(debug=True)
