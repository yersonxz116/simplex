# Solver Simplex con Groq

Aplicacion de escritorio en Python para convertir un enunciado en espanol en un modelo de programacion lineal, resolverlo con el metodo simplex y exportar el procedimiento completo a Excel.

## Funciones actuales

- Interfaz grafica con `tkinter`.
- Captura del enunciado del problema en lenguaje natural.
- Extraccion del modelo lineal usando Groq.
- Estandarizacion del modelo con variables de holgura, excedente y artificiales.
- Resolucion por metodo simplex con registro de iteraciones.
- Resumen en pantalla con:
  - formulacion del modelo general,
  - formulacion del modelo estandar,
  - variables basicas y no basicas,
  - solucion optima y valor de `Z`.
- Exportacion a `simplex_resultado.xlsx` o a una ruta elegida por el usuario.
- Guardado local de la API key en `~/.simplex_groq_solver.json`.

## Estructura del proyecto

- `simplex_groq_solver_corregido_v2.py`: script principal con UI, integracion con Groq, simplex y exportacion a Excel.
- `requirements.txt`: dependencias del proyecto.
- `simplex_resultado.xlsx`: archivo generado por la aplicacion.

## Requisitos

- Python 3.13 o compatible
- API key de Groq

Dependencias:

- `groq`
- `openpyxl`

## Instalacion

En Linux o macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuracion

Puedes escribir la API key directamente en la ventana y guardarla desde la aplicacion. Tambien puedes usar variables de entorno:

```bash
export GROQ_API_KEY="tu_api_key"
export GROQ_MODEL="llama-3.3-70b-versatile"
```

En Windows PowerShell:

```powershell
$env:GROQ_API_KEY="tu_api_key"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
```

Si existe `GROQ_API_KEY` en el sistema, esa clave tiene prioridad sobre la guardada localmente.

## Uso

Ejecuta la aplicacion:

```bash
python simplex_groq_solver_corregido_v2.py
```

Flujo de trabajo:

1. Guarda la API key de Groq si aun no esta configurada.
2. Pega el enunciado del problema.
3. Pulsa `Resolver y generar Excel`.
4. Revisa el resumen en pantalla.
5. Usa `Descargar Excel` para guardar una copia en la ruta que elijas.

## Salida

El Excel generado incluye:

- modelo general,
- modelo estandar,
- tablas simplex por iteracion,
- operaciones de Gauss,
- pivote resaltado,
- variables basicas y no basicas,
- solucion optima final.

## Verificacion rapida

Para validar sintaxis:

```bash
python -m py_compile simplex_groq_solver_corregido_v2.py
```

## Notas

- La aplicacion puede fallar si falta la API key o si Groq devuelve un modelo invalido.
- El solver detecta problemas no acotados e infactibles y muestra el error en pantalla.
- No subas claves, archivos `.env` ni `~/.simplex_groq_solver.json` al repositorio.
