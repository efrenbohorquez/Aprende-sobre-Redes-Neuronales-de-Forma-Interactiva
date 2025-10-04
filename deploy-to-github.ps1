# 🚀 Script de Deployment a GitHub

# Este script te ayudará a subir el proyecto al repositorio de GitHub

# INSTRUCCIONES:
# ==============
# 1. Abre PowerShell en la carpeta del proyecto
# 2. Ejecuta: .\deploy-to-github.ps1

Write-Host "🧠 Aprende sobre Redes Neuronales - Deployment Script" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si Git está instalado
Write-Host "📋 Verificando Git..." -ForegroundColor Yellow
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git no está instalado. Por favor instala Git primero:" -ForegroundColor Red
    Write-Host "   https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Git encontrado" -ForegroundColor Green
Write-Host ""

# Verificar si ya existe un repositorio Git
if (Test-Path .git) {
    Write-Host "📁 Repositorio Git ya existe" -ForegroundColor Yellow
    $respuesta = Read-Host "¿Deseas reinicializar el repositorio? (s/N)"
    if ($respuesta -eq 's' -or $respuesta -eq 'S') {
        Remove-Item -Recurse -Force .git
        Write-Host "✅ Repositorio reinicializado" -ForegroundColor Green
    }
}

# Inicializar repositorio si no existe
if (!(Test-Path .git)) {
    Write-Host "🔧 Inicializando repositorio Git..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Repositorio inicializado" -ForegroundColor Green
}
Write-Host ""

# Añadir archivos
Write-Host "📦 Añadiendo archivos al staging..." -ForegroundColor Yellow
git add .
Write-Host "✅ Archivos añadidos" -ForegroundColor Green
Write-Host ""

# Crear commit
Write-Host "💾 Creando commit inicial..." -ForegroundColor Yellow
git commit -m "Initial commit: Proyecto educativo de Redes Neuronales con Gradio

- Aplicación interactiva con 6 módulos educativos
- Documentación profesional completa
- Ejemplos de CNN y RNN
- Visualizaciones con Matplotlib y Plotly
- Interfaz web con Gradio"
Write-Host "✅ Commit creado" -ForegroundColor Green
Write-Host ""

# Configurar rama principal
Write-Host "🌿 Configurando rama principal..." -ForegroundColor Yellow
git branch -M main
Write-Host "✅ Rama configurada" -ForegroundColor Green
Write-Host ""

# Añadir remote
Write-Host "🔗 Configurando repositorio remoto..." -ForegroundColor Yellow
$repo_url = "https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva.git"

# Verificar si ya existe el remote
$existing_remote = git remote get-url origin 2>$null
if ($existing_remote) {
    Write-Host "📌 Remote 'origin' ya existe: $existing_remote" -ForegroundColor Yellow
    $respuesta = Read-Host "¿Deseas actualizarlo? (s/N)"
    if ($respuesta -eq 's' -or $respuesta -eq 'S') {
        git remote remove origin
        git remote add origin $repo_url
        Write-Host "✅ Remote actualizado" -ForegroundColor Green
    }
} else {
    git remote add origin $repo_url
    Write-Host "✅ Remote añadido" -ForegroundColor Green
}
Write-Host ""

# Información final
Write-Host "✨ ¡Configuración completada!" -ForegroundColor Green
Write-Host ""
Write-Host "📌 Siguientes pasos:" -ForegroundColor Cyan
Write-Host "   1. Asegúrate de haber creado el repositorio en GitHub" -ForegroundColor White
Write-Host "      URL: https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva" -ForegroundColor White
Write-Host ""
Write-Host "   2. Para subir los cambios, ejecuta:" -ForegroundColor White
Write-Host "      git push -u origin main" -ForegroundColor Yellow
Write-Host ""
Write-Host "   3. Si el repositorio ya existe con contenido:" -ForegroundColor White
Write-Host "      git pull origin main --allow-unrelated-histories" -ForegroundColor Yellow
Write-Host "      git push -u origin main" -ForegroundColor Yellow
Write-Host ""

# Preguntar si desea hacer push automáticamente
$respuesta = Read-Host "¿Deseas hacer push ahora? (s/N)"
if ($respuesta -eq 's' -or $respuesta -eq 'S') {
    Write-Host ""
    Write-Host "🚀 Subiendo archivos a GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ ¡Proyecto subido exitosamente a GitHub!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🎉 Tu proyecto ya está en línea:" -ForegroundColor Cyan
        Write-Host "   https://github.com/efrenbohorquez/Aprende-sobre-Redes-Neuronales-de-Forma-Interactiva" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "⚠️  Hubo un problema al hacer push." -ForegroundColor Red
        Write-Host "   Posibles soluciones:" -ForegroundColor Yellow
        Write-Host "   1. Verifica que el repositorio exista en GitHub" -ForegroundColor White
        Write-Host "   2. Configura tus credenciales de GitHub" -ForegroundColor White
        Write-Host "   3. Si hay contenido existente:" -ForegroundColor White
        Write-Host "      git pull origin main --allow-unrelated-histories" -ForegroundColor Yellow
        Write-Host "      git push -u origin main" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🎓 ¡Gracias por usar este proyecto educativo!" -ForegroundColor Cyan
