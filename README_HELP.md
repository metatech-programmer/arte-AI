
---

# ArteIAProyecto

Este proyecto utiliza TensorFlow y OpenCV para desarrollar una solución basada en inteligencia artificial aplicada al arte.  

### 🚨 Problemas de Compatibilidad con Python 3.12
Durante la instalación de las dependencias, algunos usuarios pueden enfrentar problemas debido a incompatibilidades entre las bibliotecas requeridas (como `numpy`) y Python 3.12. Esto puede causar errores como:

```plaintext
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
```

### Solución Recomendada
Se recomienda usar **Python 3.10** o **3.11**, ya que las bibliotecas del proyecto han sido probadas en estas versiones.

---

## 🚀 Instalación

Sigue estos pasos para configurar el entorno:

1. **Clona este repositorio**:
   ```bash
   git clone https://github.com/tu_usuario/ArteIAProyecto.git
   cd ArteIAProyecto
   ```

2. **Instala Python 3.10**:
   - Descarga Python 3.10 desde [python.org](https://www.python.org/downloads/).
   - Asegúrate de agregar Python al PATH durante la instalación.

3. **Crea un entorno virtual**:
   ```bash
   python -m venv venv
   ```

4. **Activa el entorno virtual**:
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. **Actualiza herramientas de construcción**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

6. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📄 Solución Alternativa: Forzar Binarios de Numpy

Si usas Python 3.12 y no puedes cambiar de versión, instala una versión binaria de `numpy` para evitar errores de compilación:
```bash
pip install numpy --only-binary :all:
```
Luego, instala el resto de las dependencias:
```bash
pip install -r requirements.txt
```

---

## 🐛 Problemas Comunes

### Error: `No module named 'distutils'`
Este error ocurre si las herramientas de construcción no están actualizadas. Resuélvelo ejecutando:
```bash
pip install --upgrade pip setuptools wheel
```

### Error: `subprocess-exited-with-error`
Elimina archivos temporales de instalación con:
```bash
pip cache purge
```

---
¡Espero que te sea útil! 🎉 Si necesitas personalizar algo, avísame.