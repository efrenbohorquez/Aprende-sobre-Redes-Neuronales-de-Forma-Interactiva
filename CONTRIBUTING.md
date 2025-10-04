# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir a **Aprende sobre Redes Neuronales de Forma Interactiva**! Este documento proporciona las directrices para contribuir al proyecto.

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo puedo contribuir?](#cómo-puedo-contribuir)
- [Proceso de Contribución](#proceso-de-contribución)
- [Guía de Estilo](#guía-de-estilo)
- [Reportar Bugs](#reportar-bugs)
- [Sugerir Mejoras](#sugerir-mejoras)
- [Preguntas](#preguntas)

## 📜 Código de Conducta

Este proyecto y todos los participantes están regidos por nuestro Código de Conducta. Al participar, se espera que mantengas estos estándares.

### Nuestro Compromiso

- Ser respetuoso y considerado con todos los contribuidores
- Aceptar críticas constructivas
- Centrarse en lo que es mejor para la comunidad educativa
- Mostrar empatía hacia otros miembros de la comunidad

## 🚀 ¿Cómo puedo contribuir?

Hay muchas formas de contribuir a este proyecto:

### 1. 📝 Mejorar la Documentación

- Corregir errores tipográficos o gramaticales
- Mejorar explicaciones existentes
- Añadir ejemplos adicionales
- Traducir documentación a otros idiomas
- Crear tutoriales o guías

### 2. 🐛 Reportar Bugs

- Reportar errores en el código
- Identificar problemas en la interfaz
- Señalar inconsistencias en la documentación

### 3. ✨ Añadir Nuevas Funcionalidades

- Implementar nuevos ejemplos de redes neuronales
- Añadir nuevas visualizaciones
- Mejorar la interfaz de usuario
- Optimizar el rendimiento

### 4. 🎨 Mejorar el Diseño

- Mejorar la interfaz de Gradio
- Crear mejores visualizaciones
- Diseñar gráficos más intuitivos

### 5. 🧪 Testing

- Escribir pruebas unitarias
- Reportar problemas de compatibilidad
- Validar en diferentes sistemas operativos

## 🔄 Proceso de Contribución

### Configuración del Entorno

1. **Fork el repositorio**
   ```bash
   # En GitHub, haz clic en "Fork"
   ```

2. **Clonar tu fork**
   ```bash
   git clone https://github.com/TU-USUARIO/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva.git
   cd Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva
   ```

3. **Configurar el repositorio upstream**
   ```bash
   git remote add upstream https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva.git
   ```

4. **Crear entorno virtual**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

5. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

### Trabajando en tu Contribución

1. **Crear una rama para tu feature**
   ```bash
   git checkout -b feature/nombre-descriptivo
   ```
   
   Convenciones de nombres de ramas:
   - `feature/` - Para nuevas funcionalidades
   - `bugfix/` - Para correcciones de bugs
   - `docs/` - Para cambios en documentación
   - `refactor/` - Para refactorizaciones de código
   - `test/` - Para añadir pruebas

2. **Hacer tus cambios**
   - Escribe código limpio y bien documentado
   - Sigue las convenciones de estilo del proyecto
   - Añade docstrings a funciones y clases
   - Comenta código complejo

3. **Probar tus cambios**
   ```bash
   python main.py
   ```
   - Verifica que todo funcione correctamente
   - Prueba en diferentes escenarios
   - Asegúrate de no romper funcionalidades existentes

4. **Commit de tus cambios**
   ```bash
   git add .
   git commit -m "Add: descripción clara del cambio"
   ```
   
   Formato de mensajes de commit:
   - `Add:` - Añadir nueva funcionalidad
   - `Fix:` - Corrección de bug
   - `Update:` - Actualización de funcionalidad existente
   - `Docs:` - Cambios en documentación
   - `Refactor:` - Refactorización de código
   - `Test:` - Añadir o modificar tests
   - `Style:` - Cambios de formato (sin cambios en código)

5. **Push a tu fork**
   ```bash
   git push origin feature/nombre-descriptivo
   ```

6. **Crear Pull Request**
   - Ve a GitHub y abre un Pull Request
   - Describe claramente tus cambios
   - Referencia issues relacionados si existen
   - Espera revisión y feedback

### Revisión del Pull Request

- Los mantenedores revisarán tu PR
- Pueden sugerir cambios o mejoras
- Responde a los comentarios y actualiza tu PR si es necesario
- Una vez aprobado, tu PR será merged

## 📐 Guía de Estilo

### Python

- Seguir [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Usar nombres descriptivos para variables y funciones
- Máximo 100 caracteres por línea
- Usar docstrings de Google style

Ejemplo de docstring:
```python
def mi_funcion(param1, param2):
    """
    Descripción breve de la función.
    
    Descripción más detallada si es necesario.
    
    Args:
        param1 (tipo): Descripción del parámetro 1
        param2 (tipo): Descripción del parámetro 2
    
    Returns:
        tipo: Descripción del valor de retorno
        
    Raises:
        Exception: Descripción de cuándo se lanza
    """
    pass
```

### Documentación

- Usar Markdown para archivos `.md`
- Incluir ejemplos de código cuando sea relevante
- Usar headers apropiados (H1, H2, H3)
- Añadir emojis para mejorar la legibilidad 🎨

### Commits

- Usar mensajes claros y descriptivos
- Escribir en tiempo presente: "Add feature" no "Added feature"
- Primera línea: resumen corto (máx. 50 caracteres)
- Línea en blanco
- Descripción detallada si es necesario

## 🐛 Reportar Bugs

Al reportar un bug, incluye:

1. **Título descriptivo**
2. **Pasos para reproducir**
   - Paso 1
   - Paso 2
   - Paso 3
3. **Comportamiento esperado**
4. **Comportamiento actual**
5. **Capturas de pantalla** (si aplica)
6. **Información del sistema**
   - SO: Windows/macOS/Linux
   - Versión de Python
   - Versión de las librerías principales

### Template de Bug Report

```markdown
## Descripción del Bug
Descripción clara y concisa del bug.

## Pasos para Reproducir
1. Ir a '...'
2. Hacer clic en '...'
3. Scroll down to '...'
4. Ver error

## Comportamiento Esperado
Descripción de lo que debería ocurrir.

## Comportamiento Actual
Descripción de lo que realmente ocurre.

## Capturas de Pantalla
Si aplica, añade capturas de pantalla.

## Información del Sistema
- SO: [e.g. Windows 10]
- Python: [e.g. 3.10.5]
- TensorFlow: [e.g. 2.13.0]
- Gradio: [e.g. 4.0.0]

## Contexto Adicional
Cualquier otra información relevante.
```

## 💡 Sugerir Mejoras

Al sugerir una mejora, incluye:

1. **Título claro**
2. **Descripción de la mejora**
3. **Motivación**: ¿Por qué es útil?
4. **Ejemplos**: ¿Cómo se vería?
5. **Alternativas consideradas**

### Template de Feature Request

```markdown
## ¿La feature está relacionada con un problema?
Descripción clara del problema. Ej: "Siempre me frustro cuando [...]"

## Describe la solución que te gustaría
Descripción clara y concisa de lo que quieres que ocurra.

## Describe alternativas que has considerado
Descripción de soluciones o features alternativas.

## Contexto Adicional
Añade cualquier otro contexto, capturas de pantalla, etc.
```

## ❓ Preguntas

Si tienes preguntas sobre el proyecto:

1. **Revisa la documentación** existente
2. **Busca en Issues** cerrados
3. **Abre una Discussion** en GitHub
4. **Contacta** a los mantenedores

## 🏆 Reconocimientos

Todos los contribuidores serán reconocidos en nuestro archivo de CONTRIBUTORS.md.

## 📞 Contacto

- **Issues**: [GitHub Issues](https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva/issues)
- **Discussions**: [GitHub Discussions](https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva/discussions)

---

## ¡Gracias por contribuir! 🎉

Tu ayuda hace que este proyecto sea mejor para toda la comunidad educativa.

<div align="center">

**Hecho con ❤️ para la educación en IA**

</div>
