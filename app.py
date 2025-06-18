from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats
import base64
import io
import json

app = Flask(__name__)

class AnalisisEstadistico:
    def __init__(self):
        self.datos = []
        self.es_agrupado = False
        self.es_muestral = True
        self.clases = []
        self.frecuencias = []
    
    def calcular_estadisticas_basicas(self, datos):
        """Calcula estadísticas básicas para datos desagrupados"""
        datos = np.array(datos)
        
        # Medidas de tendencia central
        media = float(np.mean(datos))
        mediana = float(np.median(datos))
        
        # Calcular moda manualmente para evitar problemas de serialización
        valores_unicos, conteos = np.unique(datos, return_counts=True)
        max_count = np.max(conteos)
        modas = valores_unicos[conteos == max_count]
        
        if len(modas) == 1:
            moda = float(modas[0])
        elif len(modas) == len(valores_unicos):
            moda = "No hay moda"
        else:
            moda = [float(m) for m in modas]
        
        # Medidas de dispersión
        if self.es_muestral:
            varianza = float(np.var(datos, ddof=1))
            desviacion_std = float(np.std(datos, ddof=1))
        else:
            varianza = float(np.var(datos, ddof=0))
            desviacion_std = float(np.std(datos, ddof=0))
        
        return {
            'media': round(media, 4),
            'mediana': round(mediana, 4),
            'moda': moda,
            'varianza': round(varianza, 4),
            'desviacion_estandar': round(desviacion_std, 4),
            'valor_minimo': round(float(np.min(datos)), 4),
            'valor_maximo': round(float(np.max(datos)), 4),
            'rango': round(float(np.max(datos) - np.min(datos)), 4)
        }
    
    def calcular_estadisticas_agrupadas(self, clases, frecuencias):
        """Calcula estadísticas para datos agrupados"""
        # Calcular puntos medios de las clases
        puntos_medios = []
        for clase in clases:
            if '-' in clase:
                limites = clase.split('-')
                punto_medio = (float(limites[0]) + float(limites[1])) / 2
                puntos_medios.append(punto_medio)
        
        puntos_medios = np.array(puntos_medios)
        frecuencias = np.array(frecuencias)
        
        # Media agrupada
        media = float(np.sum(puntos_medios * frecuencias) / np.sum(frecuencias))
        
        # Varianza agrupada
        if self.es_muestral:
            varianza = float(np.sum(frecuencias * (puntos_medios - media)**2) / (np.sum(frecuencias) - 1))
        else:
            varianza = float(np.sum(frecuencias * (puntos_medios - media)**2) / np.sum(frecuencias))
        
        desviacion_std = float(np.sqrt(varianza))
        
        # Mediana agrupada (aproximada)
        n = int(np.sum(frecuencias))
        frecuencias_acum = np.cumsum(frecuencias)
        clase_mediana_idx = int(np.where(frecuencias_acum >= n/2)[0][0])
        
        # Moda agrupada (clase con mayor frecuencia)
        clase_modal_idx = int(np.argmax(frecuencias))
        clase_modal = clases[clase_modal_idx]
        
        # Curtosis y sesgo
        momento3 = float(np.sum(frecuencias * (puntos_medios - media)**3) / np.sum(frecuencias))
        momento4 = float(np.sum(frecuencias * (puntos_medios - media)**4) / np.sum(frecuencias))
        
        sesgo = float(momento3 / (desviacion_std**3))
        curtosis = float((momento4 / (desviacion_std**4)) - 3)
        
        return {
            'media': round(media, 4),
            'mediana_aproximada': f"Clase {clase_mediana_idx + 1}: {clases[clase_mediana_idx]}",
            'moda': f"Clase modal: {clase_modal}",
            'varianza': round(varianza, 4),
            'desviacion_estandar': round(desviacion_std, 4),
            'sesgo': round(sesgo, 4),
            'curtosis': round(curtosis, 4)
        }
    
    def crear_tabla_frecuencias(self, datos):
        """Crea tabla de frecuencias para datos desagrupados"""
        valores_unicos, frecuencias = np.unique(datos, return_counts=True)
        n_total = len(datos)
        
        tabla = []
        frecuencia_acumulada = 0
        
        for valor, freq in zip(valores_unicos, frecuencias):
            frecuencia_acumulada += int(freq)
            freq_relativa = float(freq) / n_total
            freq_relativa_acum = frecuencia_acumulada / n_total
            
            tabla.append({
                'valor': float(valor),
                'frecuencia': int(freq),
                'frecuencia_relativa': round(freq_relativa, 4),
                'frecuencia_acumulada': frecuencia_acumulada,
                'frecuencia_relativa_acumulada': round(freq_relativa_acum, 4)
            })
        
        return tabla
    
    def generar_graficas(self, datos, tipo='desagrupado', clases=None, frecuencias=None):
        """Genera gráficas según el tipo de datos"""
        graficas = {}
        
        if tipo == 'desagrupado':
            # Histograma
            plt.figure(figsize=(10, 6))
            n, bins, patches = plt.hist(datos, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Histograma')
            plt.xlabel('Valores')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            
            # Detectar sesgo visual
            media = float(np.mean(datos))
            mediana = float(np.median(datos))
            if media > mediana:
                sesgo_visual = "derecha (positivo)"
            elif media < mediana:
                sesgo_visual = "izquierda (negativo)"
            else:
                sesgo_visual = "simétrico"
            
            plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')
            plt.axvline(mediana, color='green', linestyle='--', label=f'Mediana: {mediana:.2f}')
            plt.legend()
            
            # Convertir a base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
            img_buffer.seek(0)
            img_string = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            graficas['histograma'] = img_string
            graficas['sesgo_visual'] = sesgo_visual
            
            # Diagrama de caja y bigotes
            plt.figure(figsize=(8, 6))
            plt.boxplot(datos, vert=True, patch_artist=True,
                       boxprops={'facecolor': 'lightblue', 'alpha': 0.7})
            plt.title('Diagrama de Caja y Bigotes')
            plt.ylabel('Valores')
            plt.grid(True, alpha=0.3)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
            img_buffer.seek(0)
            img_string = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            graficas['boxplot'] = img_string
            
        elif tipo == 'agrupado' and clases and frecuencias:
            # Histograma para datos agrupados
            plt.figure(figsize=(10, 6))
            
            # Crear posiciones para las barras
            x_pos = range(len(clases))
            plt.bar(x_pos, frecuencias, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Histograma - Datos Agrupados')
            plt.xlabel('Clases')
            plt.ylabel('Frecuencia')
            plt.xticks(x_pos, clases, rotation=45)
            plt.grid(True, alpha=0.3)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
            img_buffer.seek(0)
            img_string = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            graficas['histograma'] = img_string
            
            # Gráfica X-R (Promedios y Rangos) - simulada para datos agrupados
            plt.figure(figsize=(12, 8))
            
            # Subgráfica 1: Gráfica X (promedios)
            plt.subplot(2, 1, 1)
            puntos_medios = []
            for clase in clases:
                if '-' in clase:
                    limites = clase.split('-')
                    punto_medio = (float(limites[0]) + float(limites[1])) / 2
                    puntos_medios.append(punto_medio)
            
            plt.plot(range(len(puntos_medios)), puntos_medios, 'bo-', linewidth=2, markersize=6)
            plt.title('Gráfica X (Promedios por Clase)')
            plt.ylabel('Valor Promedio')
            plt.grid(True, alpha=0.3)
            
            # Subgráfica 2: Gráfica R (rangos)
            plt.subplot(2, 1, 2)
            rangos = []
            for clase in clases:
                if '-' in clase:
                    limites = clase.split('-')
                    rango = float(limites[1]) - float(limites[0])
                    rangos.append(rango)
            
            plt.plot(range(len(rangos)), rangos, 'ro-', linewidth=2, markersize=6)
            plt.title('Gráfica R (Rangos por Clase)')
            plt.xlabel('Número de Clase')
            plt.ylabel('Rango')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
            img_buffer.seek(0)
            img_string = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            graficas['grafica_xr'] = img_string
        
        return graficas

# Instancia global del analizador
analizador = AnalisisEstadistico()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/configurar', methods=['POST'])
def configurar():
    data = request.get_json()
    analizador.es_muestral = data.get('es_muestral', True)
    analizador.es_agrupado = data.get('es_agrupado', False)
    
    return jsonify({'status': 'success', 'message': 'Configuración guardada'})

@app.route('/procesar_datos', methods=['POST'])
def procesar_datos():
    data = request.get_json()
    
    try:
        if analizador.es_agrupado:
            # Datos agrupados
            clases = data.get('clases', [])
            frecuencias = data.get('frecuencias', [])
            frecuencias = [int(f) for f in frecuencias]
            
            analizador.clases = clases
            analizador.frecuencias = frecuencias
            
            # Calcular estadísticas
            estadisticas = analizador.calcular_estadisticas_agrupadas(clases, frecuencias)
            
            # Generar gráficas
            graficas = analizador.generar_graficas([], tipo='agrupado', clases=clases, frecuencias=frecuencias)
            
            # Calcular parámetros adicionales
            valor_max = float(max([float(clase.split('-')[1]) for clase in clases if '-' in clase]))
            valor_min = float(min([float(clase.split('-')[0]) for clase in clases if '-' in clase]))
            rango = float(valor_max - valor_min)
            num_clases = int(len(clases))
            amplitud = float(rango / num_clases if num_clases > 0 else 0)
            
            resultado = {
                'tipo': 'agrupado',
                'estadisticas': estadisticas,
                'graficas': graficas,
                'parametros_adicionales': {
                    'valor_maximo': round(valor_max, 4),
                    'valor_minimo': round(valor_min, 4),
                    'rango': round(rango, 4),
                    'num_clases': num_clases,
                    'amplitud': round(amplitud, 4)
                }
            }
            
        else:
            # Datos desagrupados
            datos_raw = data.get('datos', [])
            datos = [float(d) for d in datos_raw]
            analizador.datos = datos
            
            # Calcular estadísticas
            estadisticas = analizador.calcular_estadisticas_basicas(datos)
            
            # Crear tabla de frecuencias
            tabla_frecuencias = analizador.crear_tabla_frecuencias(datos)
            
            # Generar gráficas
            graficas = analizador.generar_graficas(datos, tipo='desagrupado')
            
            resultado = {
                'tipo': 'desagrupado',
                'estadisticas': estadisticas,
                'tabla_frecuencias': tabla_frecuencias,
                'graficas': graficas
            }
        
        return jsonify({'status': 'success', 'resultado': resultado})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Template HTML
html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Programa de Análisis Estadístico</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .content {
            padding: 30px;
        }
        
        .step {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 5px solid #4facfe;
        }
        
        .step h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #4facfe;
        }
        
        .radio-group {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }
        
        .radio-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .radio-item input[type="radio"] {
            width: auto;
        }
        
        .btn {
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .hidden {
            display: none;
        }
        
        .results {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-top: 20px;
        }
        
        .results h3 {
            color: #333;
            margin-bottom: 20px;
            border-bottom: 2px solid #4facfe;
            padding-bottom: 10px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4facfe;
        }
        
        .stat-label {
            font-weight: 600;
            color: #666;
            font-size: 0.9em;
        }
        
        .stat-value {
            font-size: 1.2em;
            font-weight: 700;
            color: #333;
            margin-top: 5px;
        }
        
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        
        th {
            background: #4facfe;
            color: white;
            font-weight: 600;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .alert-error {
            background: #fee;
            border-left: 4px solid #e74c3c;
            color: #c0392b;
        }
        
        .alert-success {
            background: #efe;
            border-left: 4px solid #27ae60;
            color: #229954;
        }
        
        .datos-agrupados-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        
        .clase-freq-item {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .clase-freq-item input {
            flex: 1;
        }
        
        .add-clase-btn {
            background: #27ae60;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .remove-clase-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>📊 Programa de Análisis Estadístico</h1>
            <p>Análisis completo de datos con medidas de tendencia central, dispersión y gráficas</p>
        </header>
        
        <div class="content">
            <!-- Paso 1: Configuración inicial -->
            <div class="step" id="step1">
                <h3>1. Configuración del Análisis</h3>
                
                <div class="form-group">
                    <label>¿Los datos son de una muestra o población?</label>
                    <div class="radio-group">
                        <div class="radio-item">
                            <input type="radio" id="muestral" name="tipo_datos" value="muestral" checked>
                            <label for="muestral">Muestral</label>
                        </div>
                        <div class="radio-item">
                            <input type="radio" id="poblacional" name="tipo_datos" value="poblacional">
                            <label for="poblacional">Poblacional</label>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>¿Los datos están agrupados o desagrupados?</label>
                    <div class="radio-group">
                        <div class="radio-item">
                            <input type="radio" id="desagrupados" name="agrupamiento" value="desagrupados" checked>
                            <label for="desagrupados">Desagrupados</label>
                        </div>
                        <div class="radio-item">
                            <input type="radio" id="agrupados" name="agrupamiento" value="agrupados">
                            <label for="agrupados">Agrupados</label>
                        </div>
                    </div>
                </div>
                
                <button class="btn" onclick="configurarAnalisis()">Continuar</button>
            </div>
            
            <!-- Paso 2: Entrada de datos -->
            <div class="step hidden" id="step2">
                <h3>2. Ingreso de Datos</h3>
                
                <!-- Datos desagrupados -->
                <div id="datos-desagrupados">
                    <div class="form-group">
                        <label for="datos-input">Ingresa los datos separados por comas:</label>
                        <textarea id="datos-input" rows="4" placeholder="Ejemplo: 12, 15, 18, 20, 22, 25, 28, 30"></textarea>
                    </div>
                </div>
                
                <!-- Datos agrupados -->
                <div id="datos-agrupados" class="hidden">
                    <div class="form-group">
                        <label>Clases y Frecuencias:</label>
                        <div id="clases-container">
                            <div class="clase-freq-item">
                                <input type="text" placeholder="Clase (ej: 10-20)" class="clase-input">
                                <input type="number" placeholder="Frecuencia" class="freq-input">
                                <button type="button" class="remove-clase-btn" onclick="removerClase(this)">×</button>
                            </div>
                        </div>
                        <button type="button" class="add-clase-btn" onclick="agregarClase()">+ Agregar Clase</button>
                    </div>
                </div>
                
                <button class="btn" onclick="procesarDatos()">Analizar Datos</button>
            </div>
            
            <!-- Resultados -->
            <div id="loading" class="loading hidden">
                <div class="spinner"></div>
            </div>
            
            <div id="resultados" class="results hidden">
                <h3>📈 Resultados del Análisis</h3>
                <div id="contenido-resultados"></div>
            </div>
        </div>
    </div>

    <script>
        let configActual = {
            es_muestral: true,
            es_agrupado: false
        };

        function configurarAnalisis() {
            // Obtener configuración
            const tipoMuestral = document.querySelector('input[name="tipo_datos"]:checked').value === 'muestral';
            const esAgrupado = document.querySelector('input[name="agrupamiento"]:checked').value === 'agrupados';
            
            configActual.es_muestral = tipoMuestral;
            configActual.es_agrupado = esAgrupado;
            
            // Mostrar/ocultar formularios según configuración
            document.getElementById('datos-desagrupados').classList.toggle('hidden', esAgrupado);
            document.getElementById('datos-agrupados').classList.toggle('hidden', !esAgrupado);
            
            // Mostrar paso 2
            document.getElementById('step1').classList.add('hidden');
            document.getElementById('step2').classList.remove('hidden');
        }

        function agregarClase() {
            const container = document.getElementById('clases-container');
            const div = document.createElement('div');
            div.className = 'clase-freq-item';
            div.innerHTML = `
                <input type="text" placeholder="Clase (ej: 10-20)" class="clase-input">
                <input type="number" placeholder="Frecuencia" class="freq-input">
                <button type="button" class="remove-clase-btn" onclick="removerClase(this)">×</button>
            `;
            container.appendChild(div);
        }

        function removerClase(btn) {
            const container = document.getElementById('clases-container');
            if (container.children.length > 1) {
                btn.parentElement.remove();
            }
        }

        async function procesarDatos() {
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('resultados').classList.add('hidden');
            
            try {
                let datosParaEnviar = { ...configActual };
                
                if (configActual.es_agrupado) {
                    // Recopilar datos agrupados
                    const claseInputs = document.querySelectorAll('.clase-input');
                    const freqInputs = document.querySelectorAll('.freq-input');
                    
                    const clases = Array.from(claseInputs).map(input => input.value).filter(v => v);
                    const frecuencias = Array.from(freqInputs).map(input => parseInt(input.value) || 0);
                    
                    if (clases.length === 0 || frecuencias.length === 0) {
                        throw new Error('Por favor ingresa al menos una clase con su frecuencia');
                    }
                    
                    datosParaEnviar.clases = clases;
                    datosParaEnviar.frecuencias = frecuencias;
                } else {
                    // Recopilar datos desagrupados
                    const datosTexto = document.getElementById('datos-input').value;
                    if (!datosTexto.trim()) {
                        throw new Error('Por favor ingresa los datos');
                    }
                    
                    const datos = datosTexto.split(',').map(d => parseFloat(d.trim())).filter(d => !isNaN(d));
                    
                    if (datos.length === 0) {
                        throw new Error('No se encontraron datos válidos');
                    }
                    
                    datosParaEnviar.datos = datos;
                }
                
                // Configurar análisis
                await fetch('/configurar', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(datosParaEnviar)
                });
                
                // Procesar datos
                const response = await fetch('/procesar_datos', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(datosParaEnviar)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    mostrarResultados(result.resultado);
                } else {
                    throw new Error(result.message);
                }
                
            } catch (error) {
                document.getElementById('contenido-resultados').innerHTML = `
                    <div class="alert alert-error">
                        <strong>Error:</strong> ${error.message}
                    </div>
                `;
                document.getElementById('resultados').classList.remove('hidden');
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        }

        function mostrarResultados(resultado) {
            let html = '';
            
            // Mostrar estadísticas
            html += '<div class="stats-grid">';
            for (const [key, value] of Object.entries(resultado.estadisticas)) {
                const label = traducirLabel(key);
                html += `
                    <div class="stat-item">
                        <div class="stat-label">${label}</div>
                        <div class="stat-value">${value}</div>
                    </div>
                `;
            }
            html += '</div>';
            
            // Parámetros adicionales para datos agrupados
            if (resultado.parametros_adicionales) {
                html += '<h3>📏 Parámetros de Agrupación</h3>';
                html += '<div class="stats-grid">';
                for (const [key, value] of Object.entries(resultado.parametros_adicionales)) {
                    const label = traducirLabel(key);
                    html += `
                        <div class="stat-item">
                            <div class="stat-label">${label}</div>
                            <div class="stat-value">${value}</div>
                        </div>
                    `;
                }
                html += '</div>';
            }
            
            // Tabla de frecuencias para datos desagrupados
            if (resultado.tabla_frecuencias) {
                html += '<h3>📊 Tabla de Frecuencias</h3>';
                html += '<div class="table-container">';
                html += '<table>';
                html += `
                    <thead>
                        <tr>
                            <th>Valor</th>
                            <th>Frecuencia</th>
                            <th>Frecuencia Relativa</th>
                            <th>Frecuencia Acumulada</th>
                            <th>Frecuencia Relativa Acumulada</th>
                        </tr>
                    </thead>
                    <tbody>
                `;
                
                resultado.tabla_frecuencias.forEach(fila => {
                    html += `
                        <tr>
                            <td>${fila.valor}</td>
                            <td>${fila.frecuencia}</td>
                            <td>${fila.frecuencia_relativa}</td>
                            <td>${fila.frecuencia_acumulada}</td>
                            <td>${fila.frecuencia_relativa_acumulada}</td>
                        </tr>
                    `;
                });
                
                html += '</tbody></table></div>';
            }
            
            // Gráficas
            if (resultado.graficas) {
                html += '<h3>📈 Gráficas</h3>';
                
                if (resultado.graficas.histograma) {
                    html += '<div class="chart-container">';
                    html += '<h4>Histograma</h4>';
                    html += `<img src="data:image/png;base64,${resultado.graficas.histograma}" alt="Histograma">`;
                    html += '</div>';
                }
                
                if (resultado.graficas.boxplot) {
                    html += '<div class="chart-container">';
                    html += '<h4>Diagrama de Caja y Bigotes</h4>';
                    html += `<img src="data:image/png;base64,${resultado.graficas.boxplot}" alt="Boxplot">`;
                    html += '</div>';
                }
                
                if (resultado.graficas.grafica_xr) {
                    html += '<div class="chart-container">';
                    html += '<h4>Gráfica X-R (Promedios y Rangos)</h4>';
                    html += `<img src="data:image/png;base64,${resultado.graficas.grafica_xr}" alt="Gráfica X-R">`;
                    html += '</div>';
                }
                
                if (resultado.graficas.sesgo_visual) {
                    html += `
                        <div class="alert alert-success">
                            <strong>Análisis de Sesgo Visual:</strong> La distribución está sesgada hacia la ${resultado.graficas.sesgo_visual}
                        </div>
                    `;
                }
            }
            
            document.getElementById('contenido-resultados').innerHTML = html;
            document.getElementById('resultados').classList.remove('hidden');
        }

        function traducirLabel(key) {
            const traducciones = {
                'media': 'Media',
                'mediana': 'Mediana',
                'moda': 'Moda',
                'varianza': 'Varianza',
                'desviacion_estandar': 'Desviación Estándar',
                'valor_minimo': 'Valor Mínimo',
                'valor_maximo': 'Valor Máximo',
                'rango': 'Rango',
                'sesgo': 'Sesgo',
                'curtosis': 'Curtosis',
                'mediana_aproximada': 'Mediana Aproximada',
                'num_clases': 'Número de Clases',
                'amplitud': 'Amplitud de Clase'
            };
            return traducciones[key] || key;
        }

        // Agregar eventos para cambio de configuración
        document.querySelectorAll('input[name="agrupamiento"]').forEach(radio => {
            radio.addEventListener('change', function() {
                const esAgrupado = this.value === 'agrupados';
                document.getElementById('datos-desagrupados').classList.toggle('hidden', esAgrupado);
                document.getElementById('datos-agrupados').classList.toggle('hidden', !esAgrupado);
            });
        });
    </script>
</body>
</html>
"""

# Crear directorio de templates si no existe
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

# Guardar el template
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html_template.strip())

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)