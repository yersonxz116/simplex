# Solver Simplex con Groq

Aplicacion en Python para resolver problemas de programacion lineal con el metodo simplex a partir de un enunciado en espanol. El script usa Groq para extraer el modelo, ejecuta el proceso simplex y exporta las tablas del procedimiento a un archivo Excel.

## Contenido del proyecto

- `simplex_groq_solver_corregido_v2.py`: script principal con interfaz grafica, extraccion del modelo, resolucion simplex y exportacion a Excel.
- `AGENTS.md`: guia local de trabajo para este directorio.
- `simplex_resultado.xlsx`: archivo generado por la aplicacion. No forma parte del codigo fuente.

## Requisitos

- Python 3.13
- Una API key de Groq

Dependencias:

- `groq`
- `openpyxl`

## Instalacion

En PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install groq openpyxl
```

## Configuracion

La aplicacion ahora permite escribir la API key de Groq directamente en la ventana y la guarda en este archivo local del usuario:

```text
~/.simplex_groq_solver.json
```

Eso evita depender de variables de entorno cuando cambias de Windows a Linux.

Si prefieres seguir usando variable de entorno, tambien funciona:

```powershell
$env:GROQ_API_KEY="tu_api_key"
```

Opcionalmente puedes cambiar el modelo:

```powershell
$env:GROQ_MODEL="llama-3.3-70b-versatile"
```

## Uso

Ejecuta el script principal:

```powershell
python .\simplex_groq_solver_corregido_v2.py
```

La aplicacion abre una interfaz donde puedes:

1. Configurar la API key de Groq.
2. Pegar el enunciado del problema.
3. Resolverlo con Groq.
4. Generar y guardar el archivo `simplex_resultado.xlsx`.

## Salida

El programa genera un Excel con:

- El modelo lineal extraido.
- Las tablas del metodo simplex por iteracion.
- La solucion optima encontrada.

## Verificacion rapida

Para validar sintaxis del script:

```powershell
python -m py_compile .\simplex_groq_solver_corregido_v2.py
```

## Estructura actual

Este repositorio todavia es pequeno y esta organizado alrededor de un solo script. Si el proyecto crece, conviene mover la logica reutilizable a `src/` y las pruebas a `tests/`.

## Notas de seguridad

- No subas `GROQ_API_KEY` al repositorio.
- No subas `~/.simplex_groq_solver.json` ni copies esa clave a archivos versionados.
- No versionas `venv/`, archivos temporales ni salidas generadas.
# simplex
