import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification, make_circles
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns
from PIL import Image
import io
import base64

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.style.use('default')

class RedNeuronalEducativa:
    def __init__(self):
        self.model = None
        
    def crear_neurona_simple(self, entrada1, entrada2, peso1, peso2, sesgo, funcion_activacion):
        """Simula el comportamiento de una neurona artificial"""
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

# Crear instancia de la clase
red_educativa = RedNeuronalEducativa()

# Crear la interfaz de Gradio
with gr.Blocks(title="🧠 Redes Neuronales Interactivas", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Aprende sobre Redes Neuronales de Forma Interactiva
    
    Esta aplicación te ayudará a entender los conceptos fundamentales de las redes neuronales
    a través de visualizaciones interactivas y ejemplos prácticos.
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
        
        # Tab 5: Recursos Adicionales
        with gr.Tab("📚 Recursos"):
            gr.Markdown("""
            ## 📖 Aprende Más sobre Redes Neuronales
            
            ### 🎯 Conceptos Clave que has Aprendido:
            
            1. **Neurona Artificial**: Unidad básica que procesa información
            2. **Pesos y Sesgos**: Parámetros que la red ajusta durante el aprendizaje
            3. **Funciones de Activación**: Introducen no-linealidad en el modelo
            4. **Arquitectura**: El número y tamaño de las capas afecta la capacidad
            5. **Backpropagation**: Algoritmo que permite el aprendizaje automático
            
            ### 🚀 Próximos Pasos:
            
            - Experimenta con diferentes configuraciones
            - Prueba con tus propios datos
            - Explora redes más complejas (CNN, RNN)
            - Aprende sobre regularización y optimización
            
            ### 🛠️ Herramientas Utilizadas:
            
            - **Gradio**: Para la interfaz interactiva
            - **TensorFlow/Scikit-learn**: Para los modelos
            - **Matplotlib/Plotly**: Para las visualizaciones
            - **NumPy**: Para cálculos numéricos
            """)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
