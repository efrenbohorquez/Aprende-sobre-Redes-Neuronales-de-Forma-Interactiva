# 📦 Guía de Instalación Detallada

Esta guía te ayudará a instalar y configurar el proyecto **Aprende sobre Redes Neuronales de Forma Interactiva** en tu sistema.

## 📋 Tabla de Contenidos

- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación de Python](#instalación-de-python)
- [Instalación del Proyecto](#instalación-del-proyecto)
- [Configuración del Entorno Virtual](#configuración-del-entorno-virtual)
- [Instalación de Dependencias](#instalación-de-dependencias)
- [Verificación de la Instalación](#verificación-de-la-instalación)
- [Solución de Problemas](#solución-de-problemas)

---

## 💻 Requisitos del Sistema

### Requisitos Mínimos

- **Sistema Operativo**: Windows 10/11, macOS 10.14+, o Linux (Ubuntu 18.04+, Debian 10+, etc.)
- **RAM**: 4 GB mínimo (8 GB recomendado)
- **Espacio en Disco**: 2 GB libres
- **Python**: 3.8 o superior
- **Conexión a Internet**: Para descargar dependencias

### Requisitos Recomendados

- **RAM**: 8 GB o más
- **Procesador**: Dual-core 2.0 GHz o superior
- **GPU**: Opcional, pero mejora el rendimiento con TensorFlow

---

## 🐍 Instalación de Python

### Windows

#### Opción 1: Descarga Directa

1. Visita [python.org/downloads](https://www.python.org/downloads/)
2. Descarga Python 3.8 o superior
3. Ejecuta el instalador
4. ✅ **IMPORTANTE**: Marca "Add Python to PATH"
5. Haz clic en "Install Now"

#### Opción 2: Microsoft Store

```powershell
# Busca "Python" en Microsoft Store e instala
```

#### Verificar Instalación

```powershell
python --version
# Debería mostrar: Python 3.x.x
```

### macOS

#### Opción 1: Homebrew (Recomendado)

```bash
# Instalar Homebrew si no lo tienes
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Python
brew install python@3.11
```

#### Opción 2: Descarga Directa

1. Visita [python.org/downloads/macos](https://www.python.org/downloads/macos/)
2. Descarga el instalador
3. Ejecuta el archivo .pkg

#### Verificar Instalación

```bash
python3 --version
# Debería mostrar: Python 3.x.x
```

### Linux

#### Ubuntu/Debian

```bash
# Actualizar repositorios
sudo apt update

# Instalar Python y pip
sudo apt install python3 python3-pip python3-venv

# Verificar instalación
python3 --version
pip3 --version
```

#### Fedora/CentOS/RHEL

```bash
# Fedora
sudo dnf install python3 python3-pip

# CentOS/RHEL
sudo yum install python3 python3-pip
```

#### Arch Linux

```bash
sudo pacman -S python python-pip
```

---

## 📥 Instalación del Proyecto

### Paso 1: Instalar Git (si no lo tienes)

#### Windows

Descarga desde [git-scm.com](https://git-scm.com/download/win)

#### macOS

```bash
brew install git
```

#### Linux

```bash
# Ubuntu/Debian
sudo apt install git

# Fedora
sudo dnf install git
```

### Paso 2: Clonar el Repositorio

```bash
# Navega al directorio donde quieres instalar
cd ~/Desktop  # o cualquier otra ubicación

# Clona el repositorio
git clone https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva.git

# Entra al directorio
cd Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva
```

### Alternativa: Descargar ZIP

1. Ve a [GitHub](https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva)
2. Haz clic en "Code" → "Download ZIP"
3. Extrae el archivo
4. Navega al directorio extraído

---

## 🔧 Configuración del Entorno Virtual

### ¿Por qué usar un entorno virtual?

- ✅ Aísla las dependencias del proyecto
- ✅ Evita conflictos con otros proyectos
- ✅ Fácil de eliminar sin afectar el sistema

### Crear Entorno Virtual

#### Windows (PowerShell)

```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.\.venv\Scripts\Activate.ps1
```

Si obtienes error de ejecución de scripts:

```powershell
# Ejecuta como Administrador
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Luego intenta activar de nuevo
.\.venv\Scripts\Activate.ps1
```

#### Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS/Linux

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate
```

### Verificar Activación

Cuando el entorno está activo, verás `(.venv)` al inicio de tu línea de comando:

```
(.venv) C:\Users\Usuario\...\Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva>
```

### Desactivar Entorno Virtual (cuando termines)

```bash
deactivate
```

---

## 📚 Instalación de Dependencias

### Actualizar pip (Recomendado)

```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

### Instalar todas las dependencias

```bash
pip install -r requirements.txt
```

Este comando instalará:
- gradio >= 4.0.0
- numpy >= 1.21.0, < 2.2.0
- matplotlib >= 3.5.0
- tensorflow >= 2.13.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- seaborn >= 0.12.0
- plotly >= 5.17.0
- pillow >= 10.0.0

### Instalación Individual (si prefieres)

```bash
pip install gradio
pip install "numpy>=1.21.0,<2.2.0"
pip install matplotlib
pip install tensorflow
pip install scikit-learn
pip install pandas
pip install seaborn
pip install plotly
pip install pillow
```

### Tiempo de Instalación

- **Conexión rápida**: 5-10 minutos
- **Conexión lenta**: 15-30 minutos
- TensorFlow es el paquete más grande (~400 MB)

---

## ✅ Verificación de la Instalación

### Verificar Dependencias

```python
# Crea un archivo test_install.py
python -c "
import gradio as gr
import numpy as np
import matplotlib
import tensorflow as tf
import sklearn
import pandas as pd
import seaborn as sns
import plotly
from PIL import Image
print('✅ Todas las dependencias instaladas correctamente!')
print(f'Gradio: {gr.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'NumPy: {np.__version__}')
"
```

### Ejecutar la Aplicación

```bash
python main.py
```

Deberías ver:

```
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

### Abrir en el Navegador

1. La aplicación debería abrirse automáticamente
2. Si no, navega a: `http://127.0.0.1:7860`
3. Explora las diferentes pestañas

---

## 🔧 Solución de Problemas

### Problema: "Python no se reconoce como comando"

**Windows:**
```powershell
# Añadir Python al PATH manualmente
# Panel de Control → Sistema → Configuración avanzada del sistema
# Variables de entorno → Path → Editar → Nuevo
# Añadir: C:\Users\TuUsuario\AppData\Local\Programs\Python\Python3xx
```

**macOS/Linux:**
```bash
# Usar python3 en lugar de python
python3 main.py
```

### Problema: Error al activar entorno virtual en Windows

```powershell
# Ejecutar PowerShell como Administrador
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problema: Error con NumPy y TensorFlow

```bash
# Desinstalar versiones incompatibles
pip uninstall numpy -y
pip uninstall tensorflow -y

# Reinstalar con versiones específicas
pip install "numpy>=1.26.0,<2.2.0"
pip install tensorflow
```

### Problema: TensorFlow tarda mucho en cargar

✅ **Esto es normal**. La aplicación usa lazy loading para cargar TensorFlow solo cuando sea necesario.

### Problema: Error de memoria al entrenar CNN

```bash
# Reduce el número de épocas
# O cierra otras aplicaciones
```

### Problema: Puerto 7860 en uso

```bash
# La aplicación usará el siguiente puerto disponible automáticamente
# O especifica uno diferente editando main.py:
# demo.launch(server_port=7861)
```

### Problema: ModuleNotFoundError

```bash
# Asegúrate de que el entorno virtual esté activado
# Verifica que (.venv) aparezca en tu terminal

# Reinstala las dependencias
pip install -r requirements.txt
```

### Problema: Conflictos de dependencias

```bash
# Elimina el entorno virtual
rm -rf .venv  # macOS/Linux
Remove-Item -Recurse -Force .venv  # Windows

# Créalo de nuevo
python -m venv .venv
# Activa
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows
# Reinstala
pip install -r requirements.txt
```

---

## 🚀 Siguientes Pasos

1. ✅ Instalación completada
2. 📖 Lee el [README.md](README.md) para aprender a usar la aplicación
3. 🎓 Explora los tutoriales interactivos
4. 🤝 Considera [contribuir](CONTRIBUTING.md) al proyecto

---

## 📞 ¿Necesitas Ayuda?

Si sigues teniendo problemas:

1. **Revisa Issues**: [GitHub Issues](https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva/issues)
2. **Abre un Issue**: Describe tu problema con detalles (SO, versión de Python, mensajes de error)
3. **Discussions**: [GitHub Discussions](https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva/discussions)

---

<div align="center">

**¡Disfruta aprendiendo sobre Redes Neuronales! 🧠🎓**

</div>
