# TFG - Neural Network Visual Designer

[![Python application](https://github.com/guillecriado/TFG/actions/workflows/python-app.yml/badge.svg)](https://github.com/guillecriado/TFG/actions/workflows/python-app.yml)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)]()

> Una aplicación web interactiva para diseñar, entrenar y evaluar redes neuronales de forma visual e intuitiva.

## 📋 Descripción

TFG Neural Network Visual Designer es una herramienta educativa y práctica que permite a usuarios de todos los niveles diseñar arquitecturas de redes neuronales mediante una interfaz gráfica interactiva. El proyecto combina la potencia de Keras/TensorFlow con una visualización intuitiva basada en NetworkX y PyVis, permitiendo a los usuarios comprender mejor cómo funcionan las redes neuronales a nivel arquitectónico.

### Características Principales

**Sistema Completo de ML Pipeline:**
- 🗂️ Carga de datasets (Scikit-Learn integrados o archivos CSV personalizados)
- 🔍 Selección interactiva de features y targets
- 🧹 Preprocesamiento de datos (limpieza, estandarización, división train/test)
- 🏗️ Diseño visual de arquitectura de red neuronal
- 🎯 Entrenamiento con múltiples optimizadores (Adam, SGD, RMSprop)
- 📊 Evaluación con métricas específicas por tipo de problema
- 🔮 Sistema de predicción en tiempo real
- 💾 Exportación de modelos entrenados

**Visualización Interactiva:**
- Representación gráfica de la arquitectura de red con NetworkX
- Interfaz PyVis para manipulación de nodos y conexiones
- Verificación automática de conectividad completa
- Edición en tiempo real (agregar/eliminar nodos y conexiones)

## 🚧 Roadmap de Mejoras

### Fase 1: Infraestructura y DevOps ⚙️
**Estado:** En Progreso

- [x] Actualización y reorganización del repositorio
- [ ] **GitHub Actions mejoradas**
  - [ ] Tests automáticos en múltiples versiones de Python
  - [ ] Linting con flake8/black
  - [ ] Coverage reports con pytest-cov
  - [ ] Build y deploy automatizado
- [ ] **Plan de releases estructurado**
  - [ ] Versionado semántico (SemVer)
  - [ ] CHANGELOG.md automatizado
  - [ ] Tags y releases en GitHub
  - [ ] Notas de release detalladas
- [ ] **Dockerización completa**
  - [ ] Dockerfile multi-stage para optimización
  - [ ] Docker Compose para desarrollo
  - [ ] Docker Compose para producción
  - [ ] Volúmenes persistentes para modelos y datasets
  - [ ] Variables de entorno configurables

### Fase 2: Backend - Performance y Arquitecturas 🚀
**Estado:** Planificado

- [ ] **Migración de Pandas a Polars**
  - [ ] Reescritura del módulo `dataset.py`
  - [ ] Optimización de operaciones de datos
  - [ ] Tests de rendimiento comparativos
  - [ ] Documentación de mejoras de performance
  
- [ ] **Soporte para múltiples arquitecturas de redes neuronales**
  - [x] **Fully Connected (MLP)** ✅ *Ya implementado*
  - [ ] **Convolutional Neural Networks (CNN)**
    - [ ] Capas Conv2D, MaxPooling2D, Flatten
    - [ ] Soporte para datasets de imágenes
    - [ ] Visualización de filtros y feature maps
  - [ ] **Recurrent Neural Networks (RNN)**
    - [ ] LSTM layers
    - [ ] GRU layers
    - [ ] Bidirectional wrappers
    - [ ] Soporte para series temporales
  - [ ] **Transformers**
    - [ ] Multi-head attention
    - [ ] Positional encoding
    - [ ] Encoder-decoder architecture
  - [ ] **Autoencoders**
    - [ ] Vanilla autoencoders
    - [ ] Variational autoencoders (VAE)
    - [ ] Denoising autoencoders
  
- [ ] **Sistema de capas modular**
  - [ ] Interfaz de selección de tipo de capa
  - [ ] Configuración de hiperparámetros por capa
  - [ ] Validación de compatibilidad entre capas

- [ ] **Mejoras de rendimiento**
  - [ ] Caché de modelos entrenados
  - [ ] Procesamiento asíncrono de datasets grandes
  - [ ] Optimización de memoria

### Fase 3: Frontend - UX/UI Mejorado 🎨
**Estado:** Planificado

- [ ] **Sistema de temas claro/oscuro**
  - [ ] Tema claro (actual)
  - [ ] Tema oscuro con paleta de colores optimizada
  - [ ] Persistencia de preferencia en localStorage
  - [ ] Transiciones suaves entre temas
  - [ ] Switch accesible en todas las páginas
  
- [ ] **Mejoras generales de UI/UX**
  - [ ] Diseño responsive mejorado
  - [ ] Animaciones y transiciones más fluidas
  - [ ] Tooltips informativos
  - [ ] Feedback visual mejorado
  - [ ] Progress indicators más detallados
  
- [ ] **Visualización avanzada**
  - [ ] Gráficos de entrenamiento en tiempo real (loss, accuracy)
  - [ ] Visualización de pesos y activaciones
  - [ ] Matriz de confusión interactiva
  - [ ] Curvas ROC y AUC
  
- [ ] **Editor de red neuronal mejorado**
  - [ ] Drag & drop de capas
  - [ ] Templates de arquitecturas predefinidas
  - [ ] Preview de dimensiones de tensores
  - [ ] Validación en tiempo real

### Fase 4: Características Avanzadas 🎓
**Estado:** Planificado

- [ ] **Experimentación y comparación**
  - [ ] Historial de experimentos
  - [ ] Comparación de múltiples modelos
  - [ ] Grid search de hiperparámetros
  - [ ] Early stopping y callbacks personalizados
  
- [ ] **Integración con frameworks**
  - [ ] Exportación a ONNX
  - [ ] Integración con TensorBoard
  - [ ] Soporte para Transfer Learning
  - [ ] Modelos pre-entrenados populares
  
- [ ] **Datasets avanzados**
  - [ ] Soporte para imágenes (PIL/OpenCV)
  - [ ] Augmentation de datos
  - [ ] Series temporales
  - [ ] Datasets personalizados con data loaders
  
- [ ] **Documentación y educación**
  - [ ] Tutoriales interactivos
  - [ ] Ejemplos paso a paso
  - [ ] Documentación de API completa
  - [ ] Video tutoriales

## 🚀 Instalación

### Requisitos Previos

- Python 3.12+
- pip o conda
- (Opcional) Docker y Docker Compose

### Opción 1: Instalación con Conda (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/guillecriado/TFG.git
cd TFG

# Crear entorno desde el archivo de configuración
conda env create -f TFG.yml

# Activar el entorno
conda activate TFG
```

### Opción 2: Instalación con pip

```bash
# Clonar el repositorio
git clone https://github.com/guillecriado/TFG.git
cd TFG

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias (próximamente requirements.txt)
pip install flask keras tensorflow pandas numpy scikit-learn networkx pyvis
```

### Opción 3: Docker (En Desarrollo)

```bash
# Clonar el repositorio
git clone https://github.com/guillecriado/TFG.git
cd TFG

# Construir y ejecutar con Docker Compose
docker-compose up --build

# La aplicación estará disponible en http://localhost:5000
```

## 💻 Uso

### Iniciar la aplicación

```bash
# Asegúrate de tener el entorno activado
conda activate TFG  # o source venv/bin/activate

# Ejecutar la aplicación Flask
python app.py

# Abrir en el navegador
# http://localhost:5000
```

### Flujo de trabajo básico

1. **Selección de Dataset** (`/`)
   - Elige un dataset de Scikit-Learn o sube tu CSV personalizado
   
2. **Selección de Salidas** (`/output_selection`)
   - Define las columnas target para tu modelo
   
3. **Selección de Entradas** (`/input_selection`)
   - Selecciona las features que utilizará tu modelo
   
4. **Preprocesamiento** (`/data_preprocessing`)
   - Configura limpieza de datos
   - Elige método de estandarización
   - Define división train/test
   - Genera red neuronal (vacía o estándar)
   
5. **Diseño de Red** (`/network_design`)
   - Visualiza tu arquitectura de red
   - Añade o elimina nodos y conexiones
   - Verifica conectividad completa
   
6. **Entrenamiento** (modal en `/network_design`)
   - Configura hiperparámetros (epochs, optimizer, learning rate)
   - Inicia entrenamiento
   - Observa progreso
   
7. **Resultados** (`/results`)
   - Visualiza métricas de evaluación
   - Realiza predicciones
   - Guarda modelo entrenado

## 🏗️ Arquitectura del Proyecto

```
TFG/
├── app.py                          # Aplicación Flask principal
├── TFG.yml                         # Configuración del entorno Conda
├── requirements.txt                # Dependencias Python (próximamente)
├── Dockerfile                      # Configuración Docker (próximamente)
├── docker-compose.yml              # Orquestación de contenedores (próximamente)
│
├── src/
│   ├── menu/
│   │   ├── __init__.py
│   │   └── dataset.py             # Gestión y procesamiento de datasets
│   │
│   ├── neuronalNetworks/
│   │   ├── __init__.py
│   │   ├── neuronal_NetworkX.py   # Clase principal de red neuronal
│   │   └── neuronal_Network_Pyvis.py  # Visualización con PyVis
│   │
│   └── test/
│       ├── __init__.py
│       └── test_neuronal_NetworkX.py  # Tests unitarios
│
├── templates/                      # Templates HTML de Flask
│   ├── dataset_selection.html
│   ├── output_selection.html
│   ├── input_selection.html
│   ├── data_preprocessing.html
│   ├── manage_neuronal_network.html
│   └── results.html
│
├── static/
│   └── graphs/
│       └── pyvis.html             # Visualización generada de la red
│
├── uploads/                        # Directorio para datasets subidos
│
└── .github/
    └── workflows/
        └── python-app.yml         # GitHub Actions workflow
```

## 🛠️ Tecnologías

### Backend Actual
- **Python 3.12**
- **Flask 3.1** - Framework web
- **Keras 3.6** con **TensorFlow 2.19** - Construcción y entrenamiento de redes neuronales
- **Pandas 2.2** - Procesamiento de datos (migración a Polars planificada)
- **Scikit-learn 1.6** - Datasets, métricas y utilidades de ML
- **NetworkX 3.4** - Representación de grafos de red neuronal
- **PyVis 0.3** - Visualización interactiva de grafos

### Frontend Actual
- **HTML5/CSS3** - Estructura y estilos
- **JavaScript Vanilla** - Interactividad
- **Vis.js** (via PyVis) - Visualización de grafos

### Mejoras Planificadas

**Backend:**
- **Polars** - Reemplazo de Pandas para mejor rendimiento
- **FastAPI** - Considerando migración para async support
- **Pydantic** - Validación de datos
- **Redis** - Caché de modelos y sesiones

**Frontend:**
- **Sistema de temas** - Modo claro/oscuro
- **Chart.js** o **Plotly.js** - Gráficos de entrenamiento
- Posible migración a **React** o **Vue.js** (a considerar)

### DevOps
- **Docker** - Containerización
- **Docker Compose** - Orquestación
- **GitHub Actions** - CI/CD
- **pytest** - Testing framework

## 📊 Comparativa de Rendimiento: Pandas vs Polars

La migración planificada de Pandas a Polars traerá mejoras significativas:

| Operación | Pandas | Polars | Mejora |
|-----------|--------|--------|--------|
| Lectura CSV (1M rows) | ~2.5s | ~0.5s | **5x más rápido** |
| Agregaciones | Baseline | 3-10x | **Hasta 10x** |
| Filtrado | Baseline | 2-5x | **Hasta 5x** |
| Joins | Baseline | 3-7x | **Hasta 7x** |
| Uso de memoria | Baseline | -50% | **50% menos memoria** |

*Nota: Métricas aproximadas basadas en benchmarks públicos de Polars.*

**Ventajas adicionales de Polars:**
- ⚡ Procesamiento paralelo nativo
- 💾 Lazy evaluation para optimización automática
- 🔄 API consistente y expresiva
- 📈 Mejor escalabilidad con datasets grandes

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Tests con cobertura
pytest --cov=src tests/

# Tests específicos
pytest src/test/test_neuronal_NetworkX.py

# Tests en verbose mode
pytest -v
```

### Tests Actuales
- ✅ Verificación de conectividad top-down
- ✅ Verificación de conectividad bottom-up
- ✅ Eliminación de nodos y edges

### Tests Planificados
- [ ] Tests de integración para endpoints Flask
- [ ] Tests de procesamiento de datos con Polars
- [ ] Tests de generación de arquitecturas
- [ ] Tests de entrenamiento y evaluación
- [ ] Tests E2E con Selenium/Playwright

## 🤝 Contribución

Las contribuciones son bienvenidas. Este es un proyecto académico pero abierto a mejoras.

### Cómo Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Guías de Estilo

- **Python:** Seguir PEP 8
- **Commits:** Usar conventional commits (feat:, fix:, docs:, etc.)
- **Tests:** Añadir tests para nuevas funcionalidades
- **Documentación:** Actualizar README y docstrings

### Áreas de Contribución Prioritarias

1. 🐛 **Bug fixes** - Siempre bienvenidos
2. 📚 **Documentación** - Mejorar guías y ejemplos
3. 🎨 **Frontend** - Implementar temas y mejorar UX
4. 🏗️ **Arquitecturas** - Añadir soporte para CNN, RNN, etc.
5. ⚡ **Performance** - Optimizaciones y migración a Polars
6. 🧪 **Testing** - Aumentar cobertura de tests

## 📝 Plan de Releases

### Versión Actual: 0.1.0 (Beta)
- ✅ Funcionalidad básica de diseño de redes fully connected
- ✅ Soporte para clasificación y regresión
- ✅ Visualización interactiva con PyVis
- ✅ Sistema completo de ML pipeline

### v0.2.0 - Infraestructura (Q1 2025)
- GitHub Actions configuradas
- Dockerización completa
- Sistema de releases automatizado
- Tests de integración

### v0.3.0 - Performance (Q2 2025)
- Migración completa a Polars
- Benchmarks de rendimiento
- Optimizaciones de memoria
- Caché de modelos

### v0.4.0 - Frontend (Q2 2025)
- Sistema de temas claro/oscuro
- UI/UX mejorado
- Visualizaciones de entrenamiento en tiempo real
- Diseño responsive optimizado

### v1.0.0 - Arquitecturas Múltiples (Q3 2025)
- Soporte para CNN
- Soporte para RNN/LSTM
- Sistema de capas modular
- Templates de arquitecturas

### v1.1.0+ - Características Avanzadas
- Transformers
- Autoencoders
- Transfer Learning
- Experimentación avanzada

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

**Guillermo Criado**
- TFG - Universidad Politécnica de Madrid (UPM), ETSISI
- GitHub: [@guillecriado](https://github.com/guillecriado)

## 🙏 Agradecimientos

- **Scikit-Learn** - Por los datasets integrados y utilidades de ML
- **Keras/TensorFlow** - Por el framework de deep learning
- **NetworkX** - Por la representación de grafos
- **PyVis** - Por la visualización interactiva
- **Flask** - Por el framework web simple y efectivo
- **UPM - ETSISI** - Por el apoyo académico

## 📞 Contacto y Soporte

Para preguntas, sugerencias o reportar bugs:
- 🐛 **Issues:** [GitHub Issues](https://github.com/guillecriado/TFG/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/guillecriado/TFG/discussions)

## 📚 Recursos Adicionales

### Documentación
- [Keras Documentation](https://keras.io/)
- [TensorFlow Guides](https://www.tensorflow.org/guide)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)
- [Polars Documentation](https://pola-rs.github.io/polars/py-polars/html/reference/)

### Tutoriales Relacionados
- [Deep Learning Basics](https://www.deeplearning.ai/)
- [Neural Networks Explained](https://www.3blue1brown.com/topics/neural-networks)

---

⭐ Si este proyecto te resulta útil para aprender sobre redes neuronales, considera darle una estrella en GitHub!

**Estado del Proyecto:** 🚧 En desarrollo activo  
**Última Actualización:** Diciembre 2024
