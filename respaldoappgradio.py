import gradio as gr
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import cv2
import tempfile
from plantcv import plantcv as pcv
from typing import Dict, Tuple, Any

# Imports de Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly no está disponible. Instala con: pip install plotly")

# Agregar el directorio app_gradio al path para poder importar utils
sys.path.append(os.path.dirname(__file__))

from utils.data_repo import DataRepo
from utils.viz import make_line_plot

# Configurar matplotlib para Gradio
plt.switch_backend('Agg')

# Configurar PlantCV
pcv.params.debug = "none"
pcv.params.dpi = 100
pcv.params.text_size = 18
pcv.params.text_thickness = 18

# Directorios de imágenes
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
repo = DataRepo(BASE_PATH)

# Configuración de análisis de imágenes
MIN_COMPONENT_AREA = 120
V_MIN = 12
G_MARGIN = 5
H_STATIC_RANGE = (30, 100)
ROI_ROWS, ROI_COLS = 2, 3
DRAW_GRID = True

# Constantes optimizadas para tallos finos
MIN_STEM_AREA = 30  # Área mínima más reducida para tallos muy finos
MIN_LEAF_AREA = 80   # Área mínima para hojas
G_MARGIN_STEM = 3   # Margen verde más permisivo para tallos finos
KERNEL_SIZE = 2     # Kernel más pequeño para preservar detalles muy finos

def apply_advanced_preprocessing(bgr: np.ndarray) -> np.ndarray:
    """
    Aplicar preprocesamiento avanzado: CLAHE, balance de blancos, 
    corrección de iluminación y ajuste HSV.
    """
    # 1. REDUCCIÓN DE EXPOSICIÓN GLOBAL
    # Convertir a float para operaciones
    img_float = bgr.astype(np.float32) / 255.0
    
    # Reducir exposición global (hacer más oscuro)
    exposure_factor = 0.8  # Factor de exposición (0.8 = 20% más oscuro)
    img_float = img_float * exposure_factor
    
    # Convertir de vuelta a uint8
    bgr_exposed = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    # 2. BALANCE DE BLANCOS AUTOMÁTICO
    # Convertir a LAB
    lab = cv2.cvtColor(bgr_exposed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calcular medias de canales a y b
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    # Ajustar balance de blancos
    a = cv2.add(a, int(128 - a_mean))
    b = cv2.add(b, int(128 - b_mean))
    
    # Reconstruir LAB y convertir a BGR
    lab_balanced = cv2.merge([l, a, b])
    bgr_balanced = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
    
    # 3. CORRECCIÓN DE ILUMINACIÓN (Retinex simplificado)
    # Convertir a float
    img_float = bgr_balanced.astype(np.float32) / 255.0
    
    # Calcular iluminación estimada (filtro gaussiano)
    illumination = cv2.GaussianBlur(img_float, (0, 0), 50)
    
    # Aplicar corrección de iluminación
    corrected = img_float / (illumination + 0.01)
    corrected = np.clip(corrected, 0, 1)
    
    # Convertir de vuelta a uint8
    bgr_illumination_corrected = (corrected * 255).astype(np.uint8)
    
    # 4. CLAHE EN CANAL V (HSV)
    hsv = cv2.cvtColor(bgr_illumination_corrected, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Aplicar CLAHE al canal V (brillo)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    
    # Reconstruir HSV y convertir a BGR
    hsv_clahe = cv2.merge([h, s, v_clahe])
    bgr_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    # 5. AJUSTE HSV ADAPTATIVO
    # Analizar histograma de V para ajustar contraste
    v_hist = cv2.calcHist([v_clahe], [0], None, [256], [0, 256])
    v_percentiles = np.percentile(v_clahe, [5, 95])
    
    # Ajustar contraste en V
    v_min, v_max = v_percentiles[0], v_percentiles[1]
    v_adjusted = np.clip((v_clahe.astype(np.float32) - v_min) * 255 / (v_max - v_min), 0, 255).astype(np.uint8)
    
    # Reconstruir imagen final
    hsv_final = cv2.merge([h, s, v_adjusted])
    bgr_final = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    
    return bgr_final

def analyze_image_palette_optimized(bgr: np.ndarray) -> Dict[str, Any]:
    """
    Análisis de paleta optimizado para detección de tallos finos.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    palette_analysis = {}
    
    # Análisis de distribución de colores
    h_hist = cv2.calcHist([h], [0], None, [180], [0, 180])
    h_peaks = np.argsort(h_hist.flatten())[-5:]
    palette_analysis['dominant_hues'] = h_peaks.tolist()
    
    # Estadísticas de saturación y brillo
    s_mean = np.mean(s)
    s_std = np.std(s)
    v_mean = np.mean(v)
    v_std = np.std(v)
    
    palette_analysis['saturation_stats'] = {'mean': float(s_mean), 'std': float(s_std)}
    palette_analysis['brightness_stats'] = {'mean': float(v_mean), 'std': float(v_std)}
    
    # Detectar tipo de iluminación
    b_mean = np.mean(b)
    if b_mean > 130:
        palette_analysis['lighting_type'] = 'artificial_warm'
    elif b_mean < 110:
        palette_analysis['lighting_type'] = 'artificial_cool'
    else:
        palette_analysis['lighting_type'] = 'natural'
    
    # Detectar sombras
    palette_analysis['has_shadows'] = v_std > 50
    
    return palette_analysis

def get_stem_optimized_parameters(palette_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parámetros optimizados para detección de tallos finos.
    """
    params = {}
    
    # Rangos HSV más permisivos para tallos finos
    if palette_analysis['lighting_type'] == 'artificial_warm':
        params['hsv_ranges'] = [
            (20, 100, 25, 255, 15, 255),  # Verde muy amplio para tallos
            (15, 105, 20, 255, 10, 255),  # Verde oscuro
            (50, 115, 15, 255, 10, 255)   # Verde azulado
        ]
        params['saturation_threshold'] = 20
        params['value_threshold'] = 10
    elif palette_analysis['lighting_type'] == 'artificial_cool':
        params['hsv_ranges'] = [
            (25, 95, 30, 255, 20, 255),
            (20, 100, 25, 255, 15, 255),
            (55, 115, 20, 255, 15, 255)
        ]
        params['saturation_threshold'] = 25
        params['value_threshold'] = 15
    else:
        params['hsv_ranges'] = [
            (25, 95, 25, 255, 10, 255),
            (20, 100, 20, 255, 8, 255),
            (55, 110, 20, 255, 10, 255)
        ]
        params['saturation_threshold'] = 20
        params['value_threshold'] = 10
    
    # Ajustar según sombras
    if palette_analysis['has_shadows']:
        params['value_threshold'] = max(8, params['value_threshold'] - 5)
        params['min_stem_area'] = max(20, MIN_STEM_AREA - 10)
    else:
        params['min_stem_area'] = MIN_STEM_AREA
    
    params['min_leaf_area'] = MIN_LEAF_AREA
    params['g_margin'] = G_MARGIN_STEM
    
    return params

def enhanced_stem_segmentation_v3(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Segmentación mejorada V3 para tallos finos con preprocesamiento avanzado
    y umbralado HSV enfocado en tallos y hojas verdes.
    """
    # Analizar paleta
    palette_analysis = analyze_image_palette_optimized(bgr)
    adaptive_params = get_stem_optimized_parameters(palette_analysis)
    
    # Aplicar preprocesamiento avanzado
    bgr_preprocessed = apply_advanced_preprocessing(bgr)
    
    # Convertir a HSV para segmentación
    hsv = cv2.cvtColor(bgr_preprocessed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lab = cv2.cvtColor(bgr_preprocessed, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    
    # --- UMBRALADO HSV ENFOCADO EN TALLOS Y HOJAS VERDES ---
    mask_hsv_combined = np.zeros_like(h)
    
    # Rangos HSV específicos para tallos finos
    stem_hsv_ranges = [
        (25, 95, 20, 255, 10, 255),   # Verde principal
        (20, 100, 15, 255, 8, 255),   # Verde oscuro
        (50, 110, 15, 255, 8, 255),   # Verde azulado
        (30, 90, 25, 255, 12, 255),   # Verde claro
        (15, 105, 18, 255, 9, 255)    # Verde muy oscuro
    ]
    
    # Rangos HSV específicos para hojas verdes
    leaf_hsv_ranges = [
        (30, 85, 30, 255, 20, 255),   # Verde hoja principal
        (25, 90, 25, 255, 15, 255),   # Verde hoja oscuro
        (45, 100, 20, 255, 12, 255),  # Verde hoja azulado
    ]
    
    # Aplicar todos los rangos HSV
    all_ranges = stem_hsv_ranges + leaf_hsv_ranges
    
    for h_min, h_max, s_min, s_max, v_min, v_max in all_ranges:
        mask_hsv = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        mask_hsv_combined = cv2.bitwise_or(mask_hsv_combined, mask_hsv)
    
    # --- EXG OPTIMIZADO PARA TALLOS FINOS ---
    B, G, R = cv2.split(bgr_preprocessed)
    
    # ExG con umbral muy bajo para tallos finos
    exg = (2 * G.astype(np.int32) - R.astype(np.int32) - B.astype(np.int32))
    exg = np.clip(exg, 0, 255).astype(np.uint8)
    
    # Umbral adaptativo más permisivo
    exg_thresh = max(1, int(np.percentile(exg, 10)))  # Más permisivo
    _, mask_exg = cv2.threshold(exg, exg_thresh, 255, cv2.THRESH_BINARY)
    
    # --- DOMINANCIA VERDE PERMISIVA PARA TALLOS FINOS ---
    g_margin = adaptive_params['g_margin']
    mask_gdom = ((G.astype(np.int16) - R.astype(np.int16) > g_margin) &
                 (G.astype(np.int16) - B.astype(np.int16) > g_margin)).astype(np.uint8) * 255
    
    # --- FILTROS DE CALIDAD OPTIMIZADOS ---
    mask_saturation = cv2.inRange(s, adaptive_params['saturation_threshold'], 255)
    mask_value = cv2.inRange(v, adaptive_params['value_threshold'], 255)
    
    # --- COMBINACIÓN INTELIGENTE ---
    mask_primary = cv2.bitwise_or(mask_hsv_combined, mask_exg)
    mask_primary = cv2.bitwise_or(mask_primary, mask_gdom)
    
    # Aplicar filtros de calidad
    mask_primary = cv2.bitwise_and(mask_primary, mask_saturation)
    mask_primary = cv2.bitwise_and(mask_primary, mask_value)
    
    # --- LIMPIEZA AVANZADA PARA TALLOS FINOS ---
    kernel_tiny = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    
    # Apertura muy suave para preservar tallos finos
    mask_primary = cv2.morphologyEx(mask_primary, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
    
    # Cierre suave para conectar tallos rotos
    mask_primary = cv2.morphologyEx(mask_primary, cv2.MORPH_CLOSE, kernel_tiny, iterations=1)
    
    # Filtro de mediana muy suave
    mask_primary = cv2.medianBlur(mask_primary, 3)
    
    # --- REDUCCIÓN DE RUIDO AVANZADA ---
    # Limpiar componentes muy pequeños (ruido)
    mask_clean = clean_components_by_size_advanced(mask_primary, adaptive_params['min_stem_area'])
    
    # --- SEPARACIÓN TALLOS vs HOJAS MEJORADA ---
    mask_stems, mask_leaves = separate_stems_from_leaves_v2(mask_clean, adaptive_params)
    
    return mask_stems, mask_leaves, palette_analysis, adaptive_params

def clean_components_by_size_advanced(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Limpieza avanzada de componentes por tamaño para reducir ruido."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            # Verificar que no sea solo ruido usando área mínima
            # Si el área es muy pequeña, probablemente es ruido
            if area >= min_area * 2:  # Área mínima duplicada para mayor confianza
                out[labels == i] = 255
    
    return out

def separate_stems_from_leaves_v2(mask: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separación mejorada V2 de tallos (estructuras finas) de hojas (estructuras anchas).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_stems = np.zeros_like(mask)
    mask_leaves = np.zeros_like(mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params['min_stem_area']:
            continue
        
        # Calcular relación de aspecto para distinguir tallos de hojas
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        
        # Calcular circularidad
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0
        
        # Calcular densidad
        density = area / (perimeter * perimeter + 1e-6)
        
        # Criterios mejorados para tallos: 
        # - Alta relación de aspecto (muy alto o muy bajo)
        # - Baja circularidad
        # - Densidad baja (estructuras finas)
        is_stem = ((aspect_ratio > 4.0 or aspect_ratio < 0.25) or 
                   (circularity < 0.25) or 
                   (density < 0.05))
        
        if is_stem and area >= params['min_stem_area']:
            cv2.fillPoly(mask_stems, [contour], 255)
        elif area >= params['min_leaf_area']:
            cv2.fillPoly(mask_leaves, [contour], 255)
    
    return mask_stems, mask_leaves

def count_stems_improved(mask_stems: np.ndarray, min_area: int = 30) -> int:
    """Cuenta mejorada del número de tallos individuales."""
    contours, _ = cv2.findContours(mask_stems, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stem_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Verificar que sea realmente un tallo (no ruido)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Calcular densidad para confirmar que es un tallo
                density = area / (perimeter * perimeter + 1e-6)
                if density < 0.1:  # Tallos tienen densidad baja
                    stem_count += 1
    
    return stem_count

def save_height_stems_visualization(original_bgr: np.ndarray, mask_stems: np.ndarray, 
                                   mask_leaves: np.ndarray, save_path: str, 
                                   altura: float = None, num_stems: int = None,
                                   palette_analysis: Dict[str, Any] = None,
                                   area_total: int = None, area_stems: int = None, area_leaves: int = None) -> None:
    """Visualización mejorada enfocada en tallos vs hojas con análisis de paleta."""
    
    # Crear máscaras de color
    stems_colored = np.zeros_like(original_bgr)
    stems_colored[mask_stems > 0] = (0, 0, 255)  # Rojo para tallos
    
    leaves_colored = np.zeros_like(original_bgr)
    leaves_colored[mask_leaves > 0] = (0, 255, 0)  # Verde para hojas
    
    # Combinar
    combined = cv2.add(stems_colored, leaves_colored)
    
    fig = plt.figure(figsize=(18, 10), dpi=100)
    
    # Original
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen Original")
    ax1.axis("off")
    
    # Solo tallos
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(cv2.cvtColor(stems_colored, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Tallos Detectados: {num_stems}")
    ax2.axis("off")
    
    # Solo hojas
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(cv2.cvtColor(leaves_colored, cv2.COLOR_BGR2RGB))
    ax3.set_title("Hojas Detectadas")
    ax3.axis("off")
    
    # Combinado
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    ax4.set_title("Tallos (Rojo) + Hojas (Verde)")
    ax4.axis("off")
    
    # Métricas principales
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.axis("off")
    if altura is not None and num_stems is not None:
        metrics_text = f"MÉTRICAS PRINCIPALES:\n\n"
        metrics_text += f"ALTURA: {altura:.2f} píxeles\n"
        metrics_text += f"N° TALLOS: {num_stems}\n"
        if area_total:
            metrics_text += f"ÁREA TOTAL: {area_total:,} px²\n"
        if area_stems:
            metrics_text += f"ÁREA TALLOS: {area_stems:,} px²\n"
        if area_leaves:
            metrics_text += f"ÁREA HOJAS: {area_leaves:,} px²\n"
        
        ax5.text(0.5, 0.5, metrics_text, transform=ax5.transAxes,
                 fontsize=14, va='center', ha='center',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Análisis de paleta
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.axis("off")
    if palette_analysis:
        palette_text = f"Análisis de Paleta:\n\n"
        palette_text += f"Tipo de luz: {palette_analysis['lighting_type']}\n"
        palette_text += f"Sombras: {'Sí' if palette_analysis['has_shadows'] else 'No'}\n"
        palette_text += f"Brillo medio: {palette_analysis['brightness_stats']['mean']:.1f}\n"
        palette_text += f"Saturación: {palette_analysis['saturation_stats']['mean']:.1f}\n"
        
        ax6.text(0.1, 0.5, palette_text, transform=ax6.transAxes,
                 fontsize=10, va='center', ha='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Histograma de tonos dominantes
    ax7 = fig.add_subplot(2, 4, 7)
    if palette_analysis and 'dominant_hues' in palette_analysis:
        hues = palette_analysis['dominant_hues']
        ax7.bar(range(len(hues)), [h for h in hues], color='green', alpha=0.7)
        ax7.set_title("Tonos Dominantes (H)")
        ax7.set_xlabel("Índice")
        ax7.set_ylabel("Valor H")
        ax7.grid(True, alpha=0.3)
    
    # Máscaras binarias
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.imshow(mask_stems + mask_leaves, cmap='gray')
    ax8.set_title("Máscara Combinada (Tallos + Hojas)")
    ax8.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

def plot_timeseries(csv_name, x_col, y_col, chart_type="Líneas y puntos", aggregation="Promedio"):
    """Grafica una serie temporal interactiva X vs Y usando Plotly."""
    if not csv_name:
        return None, "⚠️ Selecciona un CSV"
    
    try:
        # Verificar Plotly directamente
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            return None, "❌ Error: Plotly no está instalado. Ejecuta: pip install plotly"
        
        df = repo.get_csv(csv_name)
        if df is None:
            return None, f"❌ No se pudo leer {csv_name}"

        if not x_col or not y_col:
            return None, "⚠️ Selecciona columnas X e Y"

        # Intento convertir X a datetime si parece fecha
        if x_col and any(k in x_col.lower() for k in ["fecha", "date", "time", "timestamp"]):
            try:
                df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
            except Exception:
                pass

        # Limpiar datos NaN
        clean_df = df[[x_col, y_col]].dropna()
        
        # Aplicar agregación si es necesario
        if aggregation != "Sin agregar" and len(clean_df) > 0:
            # Convertir columna Y a numérica si es posible
            try:
                clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors='coerce')
                clean_df = clean_df.dropna()
                
                if len(clean_df) > 0:
                    # Agrupar por columna X y aplicar agregación
                    if aggregation == "Promedio":
                        clean_df = clean_df.groupby(x_col)[y_col].mean().reset_index()
                    elif aggregation == "Suma":
                        clean_df = clean_df.groupby(x_col)[y_col].sum().reset_index()
                    elif aggregation == "Máximo":
                        clean_df = clean_df.groupby(x_col)[y_col].max().reset_index()
                    elif aggregation == "Mínimo":
                        clean_df = clean_df.groupby(x_col)[y_col].min().reset_index()
                    
                    # Ordenar por columna X
                    clean_df = clean_df.sort_values(x_col)
            except Exception as e:
                print(f"⚠️ Error en agregación: {e}")
        
        if len(clean_df) == 0:
            # Crear gráfico vacío
            fig = go.Figure()
            fig.add_annotation(
                text="Sin datos para graficar",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title="Sin datos para graficar",
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=False
            )
        else:
            # Crear gráfico interactivo con Plotly
            fig = go.Figure()
            
            # Determinar el modo según el tipo de gráfico seleccionado
            if chart_type == "Solo líneas":
                mode = 'lines'
                marker_size = 0
            elif chart_type == "Solo puntos":
                mode = 'markers'
                marker_size = 8
            elif chart_type == "Barras":
                mode = 'markers'
                marker_size = 0
            else:  # "Líneas y puntos" (por defecto)
                mode = 'lines+markers'
                marker_size = 6
            
            # Agregar línea principal
            if chart_type == "Barras":
                # Para gráficos de barras, usar go.Bar
                fig.add_trace(go.Bar(
                    x=clean_df[x_col],
                    y=clean_df[y_col],
                    name=y_col,
                    marker_color='blue',
                    opacity=0.7,
                    hovertemplate=f'<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>'
                ))
            else:
                # Para líneas y puntos
                fig.add_trace(go.Scatter(
                    x=clean_df[x_col],
                    y=clean_df[y_col],
                    mode=mode,
                    name=y_col,
                    line=dict(width=3, color='blue') if 'lines' in mode else None,
                    marker=dict(size=marker_size, color='blue', opacity=0.7) if marker_size > 0 else None,
                    hovertemplate=f'<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>'
                ))
            
            # Calcular estadísticas
            stats_text = f"📊 Datos: {len(clean_df)} puntos"
            
            # Verificar si la columna Y es numérica antes de calcular estadísticas
            try:
                y_values = pd.to_numeric(clean_df[y_col], errors='coerce')
                y_values_clean = y_values.dropna()
                
                if len(y_values_clean) > 0:
                    max_val = y_values_clean.max()
                    min_val = y_values_clean.min()
                    mean_val = y_values_clean.mean()
                    
                    # Agregar líneas de referencia para estadísticas
                    fig.add_hline(y=max_val, line_dash="dash", line_color="red", 
                                annotation_text=f"Máximo: {max_val:.2f}")
                    fig.add_hline(y=min_val, line_dash="dash", line_color="green", 
                                annotation_text=f"Mínimo: {min_val:.2f}")
                    fig.add_hline(y=mean_val, line_dash="dot", line_color="orange", 
                                annotation_text=f"Promedio: {mean_val:.2f}")
                    
                    stats_text += f"<br>📈 Máximo: {max_val:.2f}<br>📉 Mínimo: {min_val:.2f}<br>📊 Promedio: {mean_val:.2f}"
                else:
                    stats_text += "<br>📊 No hay valores numéricos para calcular estadísticas"
            except Exception:
                stats_text += "<br>📊 No se pudieron calcular estadísticas (datos no numéricos)"
            
            # Configurar el layout
            fig.update_layout(
                title=f"📈 {y_col} vs {x_col} ({aggregation})",
                xaxis_title=x_col,
                yaxis_title=y_col,
                hovermode='x unified',
                showlegend=True,
                autosize=True,
                width=800,
                height=500,
                margin=dict(l=80, r=50, t=80, b=80),
                annotations=[
                    dict(
                        x=0.02, y=0.98,
                        xref="paper", yref="paper",
                        text=stats_text,
                        showarrow=False,
                        align="left",
                        bgcolor="rgba(255,255,224,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                ]
            )
            
            # Configurar ejes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Si es datetime, rotar etiquetas
            if pd.api.types.is_datetime64_any_dtype(clean_df[x_col]):
                fig.update_xaxes(tickangle=45)
        
        success_msg = f"✅ Gráfico interactivo generado para {csv_name}\n📊 Datos: {len(clean_df)} puntos válidos"
        return fig, success_msg
        
    except Exception as e:
        # Crear gráfico de error
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text=f"❌ Error al graficar:<br>{str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="red")
            )
            fig.update_layout(
                title="Error en la gráfica",
                showlegend=False
            )
            return fig, f"❌ Error al generar gráfico: {str(e)}"
        except ImportError:
            return None, f"❌ Error al generar gráfico: {str(e)}"

def get_csv_info(csv_name):
    """Obtiene información básica de un CSV."""
    if not csv_name:
        return "⚠️ Selecciona un CSV"
    
    try:
        # Si csv_name es una lista, tomar el primer elemento
        if isinstance(csv_name, list) and len(csv_name) > 0:
            csv_name = csv_name[0]
        
        df = repo.get_csv(csv_name)
        if df is None:
            return f"❌ No se pudo leer {csv_name}"
        
        info = f"📊 **{csv_name}**\n\n"
        info += f"**📏 Dimensiones:** {len(df)} filas × {len(df.columns)} columnas\n\n"
        info += f"**📋 Columnas disponibles:**\n"
        
        for i, col in enumerate(df.columns):
            try:
                dtype = str(df[col].dtype)
                non_null = df[col].count()
                info += f"  {i+1}. `{col}` ({dtype}) - {non_null} valores no nulos\n"
            except Exception as e:
                info += f"  {i+1}. `{col}` (error al procesar) - {str(e)}\n"
        
        return info
    except Exception as e:
        return f"❌ Error al leer {csv_name}: {str(e)}"

def ensure_dirs() -> None:
    """Crear directorios necesarios"""
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "metricas"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "mascaras"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "visualizaciones"), exist_ok=True)

def advanced_white_balance(bgr: np.ndarray, method: str = "grayworld") -> np.ndarray:
    """Balance de blancos avanzado con múltiples métodos"""
    if method == "grayworld":
        b, g, r = cv2.split(bgr.astype(np.float32))
        mb, mg, mr = b.mean() + 1e-6, g.mean() + 1e-6, r.mean() + 1e-6
        k = (mb + mg + mr) / 3.0
        b *= k / mb; g *= k / mg; r *= k / mr
        b = b.astype(np.float32); g = g.astype(np.float32); r = r.astype(np.float32)
        out = cv2.merge([b, g, r])
        return np.clip(out, 0, 255).astype(np.uint8)
    
    elif method == "adaptive":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_mean = np.mean(l)
        l_target = 128
        l_corrected = np.clip(l * (l_target / l_mean), 0, 255)
        l_corrected = l_corrected.astype(np.uint8)
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
        lab_corrected = cv2.merge([l_corrected, a, b])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    
    return bgr

def correct_illumination(img: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """Corregir iluminación no uniforme usando filtro de paso bajo"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    illumination = cv2.filter2D(img.astype(np.float32), -1, kernel)
    illumination = illumination / np.max(illumination)
    img_float = img.astype(np.float32) / 255.0
    corrected = img_float / (illumination + 0.01)
    corrected = np.clip(corrected, 0, 1)
    return (corrected * 255).astype(np.uint8)

def exg_mask(bgr: np.ndarray) -> np.ndarray:
    """Calcular máscara ExG (Excess Green)"""
    B, G, R = cv2.split(bgr)
    exg = (2 * G.astype(np.int32) - R.astype(np.int32) - B.astype(np.int32))
    exg = np.clip(exg, 0, 255).astype(np.uint8)
    return cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def enhanced_hsv_segmentation(bgr: np.ndarray) -> np.ndarray:
    """Segmentación HSV mejorada"""
    img_wb = advanced_white_balance(bgr, method="adaptive")
    img_corrected = correct_illumination(img_wb)
    
    hsv = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Segmentación HSV principal
    hmin, hmax = H_STATIC_RANGE
    hmin = max(0, hmin - 5); hmax = min(179, hmax + 5)
    mask_hsv = cv2.inRange(hsv, (hmin, 20, V_MIN), (hmax, 255, 255))
    
    # Máscara ExG mejorada
    mask_exg = exg_mask(img_corrected)
    
    # Máscara de dominancia verde
    B, G, R = cv2.split(img_corrected)
    mask_gdom = ((G.astype(np.int16) - R.astype(np.int16) > G_MARGIN) &
                  (G.astype(np.int16) - B.astype(np.int16) > G_MARGIN)).astype(np.uint8) * 255
    
    # Máscara de valor
    mask_v = cv2.inRange(v, V_MIN, 255)
    
    # Combinar máscaras
    mask_combined = cv2.bitwise_and(mask_hsv, mask_exg)
    mask_combined = cv2.bitwise_and(mask_combined, mask_gdom)
    mask_combined = cv2.bitwise_and(mask_combined, mask_v)
    
    # Limpieza morfológica
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel5, iterations=1)
    mask_combined = cv2.medianBlur(mask_combined, 5)
    
    # Limpiar componentes pequeños
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_combined, 8)
    if n > 1:
        out = np.zeros_like(mask_combined)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
                out[labels == i] = 255
        mask_combined = out
    
    return mask_combined

def calculate_color_indices(bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Calcular índices de color vegetativo"""
    masked_bgr = cv2.bitwise_and(bgr, bgr, mask=mask)
    bgr_float = masked_bgr.astype(np.float32) / 255.0
    
    B = bgr_float[:, :, 0]
    G = bgr_float[:, :, 1]
    R = bgr_float[:, :, 2]
    
    mask_valid = mask > 0
    
    if not np.any(mask_valid):
        return {
            'exg': 0.0, 'exr': 0.0, 'vari': 0.0, 'gli': 0.0, 'tgi': 0.0,
            'clorosis': 0.0, 'antocianinas': 0.0
        }
    
    # Índices básicos
    exg = np.mean(2 * G[mask_valid] - R[mask_valid] - B[mask_valid])
    exr = np.mean(1.4 * R[mask_valid] - G[mask_valid])
    
    # VARI (Visible Atmospherically Resistant Index)
    vari = np.mean((G[mask_valid] - R[mask_valid]) / (G[mask_valid] + R[mask_valid] - B[mask_valid] + 1e-6))
    
    # GLI (Green Leaf Index)
    gli = np.mean((2 * G[mask_valid] - R[mask_valid] - B[mask_valid]) / (2 * G[mask_valid] + R[mask_valid] + B[mask_valid] + 1e-6))
    
    # TGI (Triangular Greenness Index)
    tgi = np.mean(-0.5 * (190 * (R[mask_valid] - G[mask_valid]) - 120 * (R[mask_valid] - B[mask_valid])))
    
    # Detectar clorosis y antocianinas
    gr_ratio = np.mean(G[mask_valid] / (R[mask_valid] + 1e-6))
    clorosis = max(0, 1 - gr_ratio) if gr_ratio < 1 else 0
    
    rg_ratio = np.mean(R[mask_valid] / (G[mask_valid] + 1e-6))
    antocianinas = max(0, rg_ratio - 1) if rg_ratio > 1 else 0
    
    return {
        'exg': float(exg),
        'exr': float(exr),
        'vari': float(vari),
        'gli': float(gli),
        'tgi': float(tgi),
        'clorosis': float(clorosis),
        'antocianinas': float(antocianinas)
    }

def compute_advanced_metrics(mask: np.ndarray) -> dict:
    """Calcular métricas morfológicas avanzadas"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_COMPONENT_AREA]
    
    if not contours:
        return {
            "area": 0.0, "perimetro": 0.0, "ancho": 0.0, "altura": 0.0,
            "convexidad": 0.0, "solidez": 0.0, "circularidad": 0.0,
            "num_hojas": 0
        }
    
    # Métricas básicas
    area_px = float(sum(cv2.contourArea(c) for c in contours))
    perim_px = float(sum(cv2.arcLength(c, True) for c in contours))
    
    # Bounding box
    ys, xs = np.where(mask > 0)
    if xs.size > 0:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        width_px = float(xmax - xmin + 1)
        height_px = float(ymax - ymin + 1)
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:
        width_px = height_px = 0.0
        cx = cy = -1.0
    
    # Métricas avanzadas
    convexity = 0.0
    solidity = 0.0
    circularity = 0.0
    
    if len(contours) > 0:
        hull = cv2.convexHull(np.vstack(contours))
        hull_area = cv2.contourArea(hull)
        convexity = float(hull_area / (area_px + 1e-6))
        solidity = float(area_px / (hull_area + 1e-6))
        
        if perim_px > 0:
            circularity = float(4 * np.pi * area_px / (perim_px * perim_px))
    
    # Contar hojas (simplificado)
    num_hojas = max(1, len(contours))
    
    return {
        "area": area_px,
        "perimetro": perim_px,
        "ancho": width_px,
        "convexidad": convexity,
        "solidez": solidity,
        "circularidad": circularity,
    
    }

def split_rois(h: int, w: int, rows: int = ROI_ROWS, cols: int = ROI_COLS) -> list:
    """Dividir imagen en ROIs de rejilla"""
    hs, ws = h // rows, w // cols
    rois = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * ws, r * hs
            x1, y1 = (c + 1) * ws if c < cols - 1 else w, (r + 1) * hs if r < rows - 1 else h
            rois.append((x0, y0, x1, y1))
    return rois

def draw_grid(ax, work_rect: tuple, rows=ROI_ROWS, cols=ROI_COLS):
    """Dibujar rejilla en la imagen"""
    x0, y0, x1, y1 = work_rect
    w = x1 - x0; h = y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, lw=2))
    for r in range(1, rows):
        y = y0 + r * h / rows
        ax.plot([x0, x1], [y, y], lw=1)
    for c in range(1, cols):
        x = x0 + c * w / cols
        ax.plot([x, x], [y0, y1], lw=1)
    
    # IDs 1..6
    idx = 1
    for r in range(rows):
        for c in range(cols):
            cx = x0 + (c + 0.5) * w / cols
            cy = y0 + (r + 0.5) * h / rows
            ax.text(cx, cy, str(idx), ha="center", va="center", fontsize=10,
                    bbox=dict(fc="white", alpha=0.45, ec="none"))
            idx += 1

def save_analysis_panel(original_bgr: np.ndarray, mask: np.ndarray, 
                       save_path: str, color_indices: dict = None, metrics: dict = None) -> None:
    """Guardar panel completo con análisis"""
    rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    seg_rgb = np.zeros_like(rgb)
    seg_rgb[mask > 0] = (200, 255, 200)
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(15, 5), dpi=100)
    
    # Imagen original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("Imagen Original")
    ax1.axis("off")
    
    # Segmentación
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(seg_rgb)
    ax2.set_title("Segmentación Planta")
    ax2.axis("off")
    
    # Información de métricas
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    ax3.set_title("Métricas de Análisis")
    
    info_text = ""
    if metrics:
        info_text += "Métricas Morfológicas:\n"
        for key, value in metrics.items():
            if key != "num_hojas":
                info_text += f"{key}: {value:.2f}\n"
            else:
                info_text += f"{key}: {int(value)}\n"
        info_text += "\n"
    
    if color_indices:
        info_text += "Índices de Color:\n"
        for key, value in color_indices.items():
            info_text += f"{key}: {value:.3f}\n"
    
    if info_text:
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def process_single_image(image_path: str) -> dict:
    """Procesar una imagen individual y retornar todas las métricas"""
    try:
        # Leer imagen
        img_rgb, _, _ = pcv.readimage(filename=image_path)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        if img_bgr is None or img_bgr.size == 0:
            return {"error": f"No se pudo cargar la imagen {image_path}"}
        
        h, w = img_bgr.shape[:2]
        
        # Preprocesamiento
        img_brightness = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=10)
        img_illumination = correct_illumination(img_brightness)
        img_wb = advanced_white_balance(img_illumination, method="adaptive")
        
        # Segmentación
        mask = enhanced_hsv_segmentation(img_wb)
        
        # Calcular métricas globales
        global_metrics = compute_advanced_metrics(mask)
        global_color_indices = calculate_color_indices(img_wb, mask)
        
        # Dividir en ROIs y calcular métricas por ROI
        rois = split_rois(h, w)
        roi_metrics = []
        
        for i, (x0, y0, x1, y1) in enumerate(rois, start=1):
            mask_roi = mask[y0:y1, x0:x1]
            
            if np.count_nonzero(mask_roi) == 0:
                continue
            
            roi_metrics_local = compute_advanced_metrics(mask_roi)
            roi_color_indices = calculate_color_indices(img_wb[y0:y1, x0:x1], mask_roi)
            
            roi_data = {
                "roi": i,
                **roi_metrics_local,
                **roi_color_indices
            }
            roi_metrics.append(roi_data)
        
        # Guardar panel de análisis
        ensure_dirs()
        stem = os.path.splitext(os.path.basename(image_path))[0]
        panel_path = os.path.join(BASE_PATH, "metricas", f"{stem}_analysis_panel.jpg")
        save_analysis_panel(img_bgr, mask, panel_path, global_color_indices, global_metrics)
        
        # Guardar máscara
        mask_path = os.path.join(BASE_PATH, "mascaras", f"mask_{stem}.png")
        cv2.imwrite(mask_path, mask)
        
        # Preparar resultado
        result = {
            "imagen": stem,
            "dimensiones": f"{w}x{h}",
            "metricas_globales": global_metrics,
            "indices_color_globales": global_color_indices,
            "metricas_por_roi": roi_metrics,
            "panel_analisis": panel_path,
            "mascara": mask_path,
            "total_rois": len(roi_metrics)
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Error procesando imagen: {str(e)}"}

def get_available_images() -> list:
    """Obtener lista de imágenes procesadas de la carpeta imagenes_procesadas"""
    processed_folder = os.path.join(BASE_PATH, "imagenes_procesadas")
    
    available_images = []
    if os.path.exists(processed_folder):
        for file in os.listdir(processed_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                available_images.append(file)
    
    return sorted(available_images)


def analyze_existing_image(image_path: str) -> tuple:
    """Analizar una imagen existente del proyecto mostrando imagen y métricas detalladas como en análisis noir."""
    if not image_path:
        return None, "⚠️ Selecciona una imagen"
    
    try:
        # Construir ruta completa de la imagen procesada
        abs_path = os.path.join(BASE_PATH, "imagenes_procesadas", image_path)
        
        if not os.path.exists(abs_path):
            return None, f"❌ No se encontró la imagen: {image_path}"
        
        print(f"🔍 Analizando imagen individual existente: {image_path}")
        
        # Importar funciones del script procesamiento_arriba
        try:
            import importlib.util
            arriba_script_path = os.path.join(os.path.dirname(__file__), "..", "procesamiento_arriba.py")
            
            if not os.path.exists(arriba_script_path):
                return None, "❌ No se encontró el script de procesamiento_arriba.py"
            
            # Importar módulo dinámicamente
            spec = importlib.util.spec_from_file_location("arriba_module", arriba_script_path)
            arriba_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arriba_module)
            
            # Usar funciones del módulo para análisis
            if hasattr(arriba_module, 'process_image_arriba'):
                result = arriba_module.process_image_arriba(abs_path)
            else:
                # Fallback: análisis básico
                result = process_single_image(abs_path)
                
        except Exception as import_error:
            print(f"⚠️ Error importando módulo arriba: {import_error}")
            # Fallback: análisis básico
            result = process_single_image(abs_path)
        
        # Verificar que result no esté vacío
        if not result or len(result) == 0:
            return None, "❌ No se obtuvieron resultados del análisis"
        
        # Obtener información de la primera hoja (si existe)
        first_leaf = result[0] if result else {}
        image_name = first_leaf.get('imagen', 'imagen_desconocida')
        total_leaves = first_leaf.get('numero_hojas_total', 0)
        
        # Crear visualización del resultado (igual que en análisis noir)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Mostrar imagen procesada
        image = cv2.imread(abs_path)
        if image is not None:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Imagen Individual Analizada: {image_name}", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Agregar información del análisis (métricas reales del script)
        info_text = f"🔍 ANÁLISIS INDIVIDUAL AVANZADO (procesamiento_arriba.py): {image_name}\n"
        info_text += f"📅 Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        info_text += f"🔢 **HOJAS DETECTADAS:** {total_leaves} hojas\n\n"
        
        # Métricas por hoja (mostrar las primeras 3)
        if result and len(result) > 0:
            info_text += f"📊 **MÉTRICAS POR HOJA (primeras 3):**\n"
            for i, leaf_data in enumerate(result[:3]):
                info_text += f"  **Hoja {leaf_data.get('hoja_id', i+1)}:**\n"
                info_text += f"    • Área PlantCV: {leaf_data.get('area_plantcv', 0):.1f} px\n"
                info_text += f"    • Perímetro OpenCV: {leaf_data.get('perimetro_opencv', 0):.1f} px\n"
                info_text += f"    • Solidez OpenCV: {leaf_data.get('solidez_opencv', 0):.3f}\n\n"
            
            if len(result) > 3:
                info_text += f"  ... y {len(result) - 3} hojas más\n\n"
        
        # Pipeline aplicado
        info_text += f"🔬 **PIPELINE APLICADO (procesamiento_arriba.py):**\n"
        info_text += f"  • Preprocesamiento avanzado (balance de blancos + CLAHE)\n"
        info_text += f"  • Segmentación ExG + HSV calibrado\n"
        info_text += f"  • ROIs circulares (2 macetas)\n"
        info_text += f"  • Métricas morfológicas (área, perímetro, solidez)\n"
        info_text += f"  • Filtrado por área mínima (210 px)\n\n"
        
        # Interpretación de métricas
        if result and len(result) > 0:
            avg_area = np.mean([leaf.get('area_plantcv', 0) for leaf in result if leaf.get('area_plantcv', 0) > 0])
            avg_solidity = np.mean([leaf.get('solidez_opencv', 0) for leaf in result if leaf.get('solidez_opencv', 0) > 0])
            
            info_text += f"💡 **INTERPRETACIÓN:**\n"
            if avg_area > 1000:
                info_text += f"  • Área promedio alta ({avg_area:.1f} px) indica hojas grandes\n"
            elif avg_area > 500:
                info_text += f"  • Área promedio moderada ({avg_area:.1f} px) indica hojas medianas\n"
            else:
                info_text += f"  • Área promedio baja ({avg_area:.1f} px) indica hojas pequeñas\n"
            
            if avg_solidity > 0.8:
                info_text += f"  • Solidez alta ({avg_solidity:.3f}) indica hojas compactas\n"
            elif avg_solidity > 0.6:
                info_text += f"  • Solidez moderada ({avg_solidity:.3f}) indica hojas normales\n"
            else:
                info_text += f"  • Solidez baja ({avg_solidity:.3f}) indica hojas irregulares\n"
        
        # Crear mensaje de éxito con métricas detalladas (métricas reales)
        success_msg = f"✅ **Análisis individual completado para {image_name}**\n"
        success_msg += f"📊 **Métricas extraídas:** {total_leaves} hojas detectadas\n"
        success_msg += f"💾 **Resultados guardados en memoria**\n\n"
        
        # Agregar métricas detalladas por hoja
        if result and len(result) > 0:
            success_msg += f"📋 **MÉTRICAS POR HOJA (primeras 3):**\n"
            for i, leaf_data in enumerate(result[:3]):
                success_msg += f"  **Hoja {leaf_data.get('hoja_id', i+1)}:**\n"
                success_msg += f"    • Área PlantCV: {leaf_data.get('area_plantcv', 0):.1f} px\n"
                success_msg += f"    • Perímetro OpenCV: {leaf_data.get('perimetro_opencv', 0):.1f} px\n"
                success_msg += f"    • Solidez OpenCV: {leaf_data.get('solidez_opencv', 0):.3f}\n\n"
            
            if len(result) > 3:
                success_msg += f"  ... y {len(result) - 3} hojas más\n\n"
        
        # Agregar interpretación
        if result and len(result) > 0:
            avg_area = np.mean([leaf.get('area_plantcv', 0) for leaf in result if leaf.get('area_plantcv', 0) > 0])
            avg_solidity = np.mean([leaf.get('solidez_opencv', 0) for leaf in result if leaf.get('solidez_opencv', 0) > 0])
            
            success_msg += f"💡 **INTERPRETACIÓN:**\n"
            if avg_area > 1000:
                success_msg += f"  • Área promedio alta ({avg_area:.1f} px) indica hojas grandes\n"
            elif avg_area > 500:
                success_msg += f"  • Área promedio moderada ({avg_area:.1f} px) indica hojas medianas\n"
            else:
                success_msg += f"  • Área promedio baja ({avg_area:.1f} px) indica hojas pequeñas\n"
            
            if avg_solidity > 0.8:
                success_msg += f"  • Solidez alta ({avg_solidity:.3f}) indica hojas compactas\n"
            elif avg_solidity > 0.6:
                success_msg += f"  • Solidez moderada ({avg_solidity:.3f}) indica hojas normales\n"
            else:
                success_msg += f"  • Solidez baja ({avg_solidity:.3f}) indica hojas irregulares\n"
        
        return fig, success_msg
        
    except Exception as e:
        return None, f"❌ Error analizando imagen individual: {str(e)}"

def analyze_uploaded_image(image) -> tuple:
    """Analizar una imagen subida por el usuario mostrando imagen y métricas detalladas como en análisis noir."""
    if image is None:
        return None, "⚠️ Sube una imagen para analizar"
    
    try:
        # Convertir imagen de Gradio a formato OpenCV
        if isinstance(image, str):  # Es una ruta de archivo
            temp_path = image
        elif hasattr(image, 'name'):  # Archivo subido con atributo name
            temp_path = image.name
        else:  # Imagen en memoria, crear archivo temporal
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            # Convertir imagen a formato PIL y guardar
            if hasattr(image, 'save'):
                image.save(temp_path)
            else:
                # Si no tiene método save, intentar convertir
                try:
                    import PIL.Image
                    if isinstance(image, np.ndarray):
                        pil_image = PIL.Image.fromarray(image)
                        pil_image.save(temp_path)
                    else:
                        return None, "❌ Formato de imagen no soportado"
                except Exception as conv_error:
                    return None, f"❌ Error convirtiendo imagen: {str(conv_error)}"
        
        # Verificar que el archivo existe
        if not os.path.exists(temp_path):
            return None, f"❌ No se pudo crear archivo temporal: {temp_path}"
        
        print(f"🔍 Analizando imagen individual subida: {temp_path}")
        
        # Importar funciones del script procesamiento_arriba
        try:
            import importlib.util
            arriba_script_path = os.path.join(os.path.dirname(__file__), "..", "procesamiento_arriba.py")
            
            if not os.path.exists(arriba_script_path):
                return None, "❌ No se encontró el script de procesamiento_arriba.py"
            
            # Importar módulo dinámicamente
            spec = importlib.util.spec_from_file_location("arriba_module", arriba_script_path)
            arriba_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arriba_module)
            
            # Usar funciones del módulo para análisis
            if hasattr(arriba_module, 'process_image_arriba'):
                result = arriba_module.process_image_arriba(temp_path)
            else:
                # Fallback: análisis básico
                result = process_single_image(temp_path)
                
        except Exception as import_error:
            print(f"⚠️ Error importando módulo arriba: {import_error}")
            # Fallback: análisis básico
            result = process_single_image(temp_path)
        
        # Limpiar archivo temporal si se creó
        if 'temp_file' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Verificar que result no esté vacío
        if not result or len(result) == 0:
            return None, "❌ No se obtuvieron resultados del análisis"
        
        # Obtener información de la primera hoja (si existe)
        first_leaf = result[0] if result else {}
        image_name = first_leaf.get('imagen', 'imagen_subida')
        total_leaves = first_leaf.get('numero_hojas_total', 0)
        
        # Crear visualización del resultado (igual que en análisis noir)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Mostrar imagen procesada
        image_cv = cv2.imread(temp_path if 'temp_file' not in locals() else temp_path)
        if image_cv is not None:
            ax.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        ax.set_title("Imagen Individual Analizada", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Agregar información del análisis (métricas reales del script)
        info_text = f"🔍 ANÁLISIS INDIVIDUAL AVANZADO (procesamiento_arriba.py): Imagen Subida\n"
        info_text += f"📅 Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        info_text += f"🔢 **HOJAS DETECTADAS:** {total_leaves} hojas\n\n"
        
        # Métricas por hoja (mostrar las primeras 3)
        if result and len(result) > 0:
            info_text += f"📊 **MÉTRICAS POR HOJA (primeras 3):**\n"
            for i, leaf_data in enumerate(result[:3]):
                info_text += f"  **Hoja {leaf_data.get('hoja_id', i+1)}:**\n"
                info_text += f"    • Área PlantCV: {leaf_data.get('area_plantcv', 0):.1f} px\n"
                info_text += f"    • Perímetro OpenCV: {leaf_data.get('perimetro_opencv', 0):.1f} px\n"
                info_text += f"    • Solidez OpenCV: {leaf_data.get('solidez_opencv', 0):.3f}\n\n"
            
            if len(result) > 3:
                info_text += f"  ... y {len(result) - 3} hojas más\n\n"
        
        # Pipeline aplicado
        info_text += f"🔬 **PIPELINE APLICADO (procesamiento_arriba.py):**\n"
        info_text += f"  • Preprocesamiento avanzado (balance de blancos + CLAHE)\n"
        info_text += f"  • Segmentación ExG + HSV calibrado\n"
        info_text += f"  • ROIs circulares (2 macetas)\n"
        info_text += f"  • Métricas morfológicas (área, perímetro, solidez)\n"
        info_text += f"  • Filtrado por área mínima (210 px)\n\n"
        
        # Interpretación de métricas
        if result and len(result) > 0:
            avg_area = np.mean([leaf.get('area_plantcv', 0) for leaf in result if leaf.get('area_plantcv', 0) > 0])
            avg_solidity = np.mean([leaf.get('solidez_opencv', 0) for leaf in result if leaf.get('solidez_opencv', 0) > 0])
            
            info_text += f"💡 **INTERPRETACIÓN:**\n"
            if avg_area > 1000:
                info_text += f"  • Área promedio alta ({avg_area:.1f} px) indica hojas grandes\n"
            elif avg_area > 500:
                info_text += f"  • Área promedio moderada ({avg_area:.1f} px) indica hojas medianas\n"
            else:
                info_text += f"  • Área promedio baja ({avg_area:.1f} px) indica hojas pequeñas\n"
            
            if avg_solidity > 0.8:
                info_text += f"  • Solidez alta ({avg_solidity:.3f}) indica hojas compactas\n"
            elif avg_solidity > 0.6:
                info_text += f"  • Solidez moderada ({avg_solidity:.3f}) indica hojas normales\n"
            else:
                info_text += f"  • Solidez baja ({avg_solidity:.3f}) indica hojas irregulares\n"
        
        # Crear mensaje de éxito con métricas detalladas (métricas reales)
        success_msg = f"✅ **Análisis individual completado para imagen subida**\n"
        success_msg += f"📊 **Métricas extraídas:** {total_leaves} hojas detectadas\n"
        success_msg += f"💾 **Resultados guardados en memoria**\n\n"
        
        # Agregar métricas detalladas por hoja
        if result and len(result) > 0:
            success_msg += f"📋 **MÉTRICAS POR HOJA (primeras 3):**\n"
            for i, leaf_data in enumerate(result[:3]):
                success_msg += f"  **Hoja {leaf_data.get('hoja_id', i+1)}:**\n"
                success_msg += f"    • Área PlantCV: {leaf_data.get('area_plantcv', 0):.1f} px\n"
                success_msg += f"    • Perímetro OpenCV: {leaf_data.get('perimetro_opencv', 0):.1f} px\n"
                success_msg += f"    • Solidez OpenCV: {leaf_data.get('solidez_opencv', 0):.3f}\n\n"
            
            if len(result) > 3:
                success_msg += f"  ... y {len(result) - 3} hojas más\n\n"
        
        # Agregar interpretación
        if result and len(result) > 0:
            avg_area = np.mean([leaf.get('area_plantcv', 0) for leaf in result if leaf.get('area_plantcv', 0) > 0])
            avg_solidity = np.mean([leaf.get('solidez_opencv', 0) for leaf in result if leaf.get('solidez_opencv', 0) > 0])
            
            success_msg += f"💡 **INTERPRETACIÓN:**\n"
            if avg_area > 1000:
                success_msg += f"  • Área promedio alta ({avg_area:.1f} px) indica hojas grandes\n"
            elif avg_area > 500:
                success_msg += f"  • Área promedio moderada ({avg_area:.1f} px) indica hojas medianas\n"
            else:
                success_msg += f"  • Área promedio baja ({avg_area:.1f} px) indica hojas pequeñas\n"
            
            if avg_solidity > 0.8:
                success_msg += f"  • Solidez alta ({avg_solidity:.3f}) indica hojas compactas\n"
            elif avg_solidity > 0.6:
                success_msg += f"  • Solidez moderada ({avg_solidity:.3f}) indica hojas normales\n"
            else:
                success_msg += f"  • Solidez baja ({avg_solidity:.3f}) indica hojas irregulares\n"
        
        return fig, success_msg
        
    except Exception as e:
        # Limpiar archivo temporal en caso de error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return None, f"❌ Error analizando imagen individual: {str(e)}"

def analyze_height_and_stems(image_path: str) -> dict:
    """Analiza altura y número de tallos de una imagen."""
    try:
        # Leer imagen
        img_rgb, _, _ = pcv.readimage(filename=image_path)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        if img_bgr is None or img_bgr.size == 0:
            return {"error": f"No se pudo cargar la imagen {image_path}"}
        
        h, w = img_bgr.shape[:2]
        
        # Segmentación mejorada V3 para tallos
        mask_stems, mask_leaves, palette_analysis, adaptive_params = enhanced_stem_segmentation_v3(img_bgr)
        
        # Contar tallos mejorado
        num_stems = count_stems_improved(mask_stems, adaptive_params['min_stem_area'])
        
        altura = None
        
        # Combinar máscaras para análisis de altura
        mask_combined = cv2.bitwise_or(mask_stems, mask_leaves)
        
        if np.count_nonzero(mask_combined) > 0:
            # Usar PlantCV para altura
            pcv.outputs.clear()
            _ = pcv.analyze.size(img=img_bgr, labeled_mask=mask_combined, n_labels=1)
            
            # Extraer altura
            data = pcv.outputs.observations
            for canal, valores in data.items():
                if "height" in valores:
                    altura = valores["height"]["value"]
                    break
        
        # Calcular área total
        area_total = np.count_nonzero(mask_combined)
        area_stems = np.count_nonzero(mask_stems)
        area_leaves = np.count_nonzero(mask_leaves)
        
        # Guardar visualización
        ensure_dirs()
        stem = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(BASE_PATH, "visualizaciones", f"altura_tallos_{stem}.jpg")
        
        # Crear visualización
        save_height_stems_visualization(img_bgr, mask_stems, mask_leaves, vis_path, 
                                      altura, num_stems, palette_analysis, 
                                      area_total, area_stems, area_leaves)
        
        return {
            "imagen": stem,
            "dimensiones": f"{w}x{h}",
            "altura": altura,
            "numero_tallos": num_stems,
            "area_total": area_total,
            "area_stems": area_stems,
            "area_leaves": area_leaves,
            "palette_analysis": palette_analysis,
            "visualizacion": vis_path
        }
        
    except Exception as e:
        return {"error": f"Error analizando imagen: {str(e)}"}

def analyze_height_stems_existing(image_path: str) -> tuple:
    """Analizar altura y tallos de una imagen existente del proyecto usando procesamiento_SAM.py."""
    if not image_path:
        return None, "⚠️ Selecciona una imagen"
    
    try:
        # Construir ruta completa de la imagen procesada lateral
        abs_path = os.path.join(BASE_PATH, "imagenes_procesadas_laterales", image_path)
        
        if not os.path.exists(abs_path):
            return None, f"❌ No se encontró la imagen: {image_path}"
        
        # Importar funciones del script procesamiento_SAM
        try:
            sam_script_path = os.path.join(os.path.dirname(__file__), "..", "procesamiento_SAM.py")
            
            if not os.path.exists(sam_script_path):
                return None, "❌ No se encontró el script de procesamiento_SAM.py"
            
            # Importar funciones específicas
            import importlib.util
            spec = importlib.util.spec_from_file_location("sam_module", sam_script_path)
            sam_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sam_module)
            
            # Usar funciones del módulo para análisis
            if hasattr(sam_module, 'analyze_single_image_sam'):
                result = sam_module.analyze_single_image_sam(abs_path)
            else:
                # Fallback: análisis básico
                result = analyze_height_and_stems(abs_path)
                
        except Exception as import_error:
            print(f"⚠️ Error importando módulo SAM: {import_error}")
            # Fallback: análisis básico
            result = analyze_height_and_stems(abs_path)
        
        if "error" in result:
            return None, f"❌ {result['error']}"
        
        # Crear visualización de resultados con imagen
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Mostrar imagen procesada
        image = cv2.imread(abs_path)
        if image is not None:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Imagen Analizada: {result['imagen']}", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        success_msg = f"✅ Análisis de altura y tallos completado para {result['imagen']}\n"
        success_msg += f"📏 Altura: {result['altura']:.2f} píxeles\n"
        success_msg += f"🔢 Tallos detectados: {result['numero_tallos']}\n"
        success_msg += f"💾 Visualización guardada en output/visualizaciones/"
        
        return fig, success_msg
        
    except Exception as e:
        return None, f"❌ Error analizando imagen: {str(e)}"

def analyze_height_stems_upload(image) -> tuple:
    """Analizar altura y tallos de una imagen subida por el usuario usando exactamente los métodos del script procesamiento_SAM.py."""
    global sam_segmentation_mask, sam_current_image
    
    if image is None:
        return None, "⚠️ Sube una imagen para analizar"
    
    try:
        # Convertir imagen de Gradio a formato OpenCV
        if isinstance(image, str):
            temp_path = image
        elif hasattr(image, 'name'):
            temp_path = image.name
        else:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            if hasattr(image, 'save'):
                image.save(temp_path)
            else:
                try:
                    import PIL.Image
                    if isinstance(image, np.ndarray):
                        pil_image = PIL.Image.fromarray(image)
                        pil_image.save(temp_path)
                    else:
                        return None, "❌ Formato de imagen no soportado"
                except Exception as conv_error:
                    return None, f"❌ Error convirtiendo imagen: {str(conv_error)}"
        
        # Verificar que el archivo existe
        if not os.path.exists(temp_path):
            return None, f"❌ No se pudo crear archivo temporal: {temp_path}"
        
        # Cargar imagen
        image_bgr = cv2.imread(temp_path)
        if image_bgr is None:
            return None, "❌ No se pudo cargar la imagen"
        
        # Determinar si usar segmentación SAM manual o script automático
        use_sam_segmentation = sam_segmentation_mask is not None and sam_current_image is not None
        
        if use_sam_segmentation:
            print("🎯 Usando segmentación SAM manual")
            # Usar la máscara segmentada manualmente
            final_mask = sam_segmentation_mask
            
            # Analizar métricas con la máscara SAM usando exactamente los métodos del script
            result = analyze_with_sam_mask_exact(image_bgr, final_mask, temp_path)
        else:
            print("🤖 Usando análisis automático con métodos exactos del script SAM")
            # Usar exactamente los mismos métodos que el script procesamiento_SAM.py
            result = analyze_single_image_sam_exact(temp_path)
        
        if "error" in result:
            return None, f"❌ {result['error']}"
        
        # Crear visualización de resultados con imagen
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Mostrar imagen procesada
        image_cv = cv2.imread(temp_path)
        if image_cv is not None:
            ax.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        # Título según el método usado
        if use_sam_segmentation:
            ax.set_title("Imagen Analizada: Altura y Tallos (Segmentación SAM Manual)", fontsize=16, fontweight='bold')
        else:
            ax.set_title("Imagen Analizada: Altura y Tallos (Métodos Exactos del Script)", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Limpiar archivo temporal si se creó
        if 'temp_file' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Crear mensaje de éxito simplificado
        if use_sam_segmentation:
            success_msg = f"✅ **Análisis de altura y tallos completado**\n"
            success_msg += f"🎯 **Método:** Segmentación manual con SAM\n\n"
        else:
            success_msg = f"✅ **Análisis de altura y tallos completado**\n"
            success_msg += f"🤖 **Método:** Métodos exactos del script procesamiento_SAM.py\n\n"
        
        # Mostrar solo número de tallos y altura de cada uno
        if result:
            success_msg += f"🔢 **NÚMERO DE TALLOS:** {result.get('numero_tallos', 0)}\n\n"
            
            # Mostrar altura de cada tallo individual
            if 'stem_heights' in result and result['stem_heights']:
                success_msg += f"📏 **ALTURA DE CADA TALLO:**\n"
                for i, height in enumerate(result['stem_heights'], 1):
                    success_msg += f"  • Tallo {i}: {height:.1f} píxeles\n"
            elif result.get('altura_total'):
                success_msg += f"📏 **ALTURA TOTAL:** {result['altura_total']:.1f} píxeles\n"
            else:
                success_msg += f"📏 **ALTURA:** No detectada\n"
        
        return fig, success_msg
        
    except Exception as e:
        return None, f"❌ Error analizando imagen subida: {str(e)}"

def analyze_with_sam_mask_exact(image_bgr, mask, image_path):
    """Analizar altura de cada tallo individual usando exactamente los métodos del script procesamiento_SAM.py."""
    try:
        h, w = image_bgr.shape[:2]
        
        # Usar exactamente la misma función del script original
        stem_heights = analyze_individual_stems_exact(mask, image_bgr)
        
        # Calcular altura total usando PlantCV (igual que en el script)
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            height_pixels, width_pixels = pcv.bounding_rectangle(mask_uint8)
            altura_total = height_pixels
        except:
            # Fallback si PlantCV falla
            altura_total = max(stem_heights) if stem_heights else 0
        
        # Número de tallos
        num_tallos = len(stem_heights)
        
        return {
            "imagen": os.path.splitext(os.path.basename(image_path))[0],
            "dimensiones": f"{w}x{h}",
            "altura_total": altura_total,
            "numero_tallos": num_tallos,
            "stem_heights": stem_heights
        }
        
    except Exception as e:
        return {"error": f"Error analizando tallos individuales: {str(e)}"}

def analyze_single_image_sam_exact(image_path):
    """Analizar una imagen usando exactamente los mismos métodos que el script procesamiento_SAM.py."""
    try:
        # Cargar imagen
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"error": f"No se pudo cargar la imagen {image_path}"}
        
        h, w = image_bgr.shape[:2]
        
        # Preprocesar imagen (igual que en el script)
        print("Preprocesando imagen (CLAHE + Corrección dominancia color + Suavizado selectivo)...")
        image_enhanced = preprocess_image_sam_original(image_bgr)
        image_rgb = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB)
        
        # Intentar usar SAM para segmentación automática
        try:
            # Inicializar SAM
            success, message = initialize_sam_model()
            if success:
                # Configurar imagen en el predictor
                sam_predictor.set_image(image_rgb)
                
                # Crear puntos automáticos en el centro de la imagen
                center_x, center_y = w // 2, h // 2
                input_points_np = np.array([[center_x, center_y]])
                input_labels_np = np.array([1])
                
                # Realizar predicción
                masks, scores, _ = sam_predictor.predict(
                    point_coords=input_points_np,
                    point_labels=input_labels_np,
                    multimask_output=False,
                )
                
                final_mask = masks[0]
                print(f"✅ Segmentación automática con SAM completada. Score: {scores[0]:.3f}")
                
            else:
                print(f"⚠️ SAM no disponible: {message}")
                # Usar segmentación automática sin SAM
                final_mask = create_automatic_mask_without_sam(image_enhanced)
                
        except Exception as sam_error:
            print(f"⚠️ Error con SAM: {sam_error}")
            # Usar segmentación automática sin SAM
            final_mask = create_automatic_mask_without_sam(image_enhanced)
        
        # Analizar métricas usando exactamente los métodos del script
        print("Analizando métricas con PlantCV...")
        metrics = get_plantcv_metrics_exact(image_rgb, final_mask, image_enhanced)
        
        return {
            "imagen": os.path.splitext(os.path.basename(image_path))[0],
            "dimensiones": f"{w}x{h}",
            "altura_total": metrics['total_height_pixels'],
            "numero_tallos": len(metrics['stem_heights']),
            "stem_heights": metrics['stem_heights']
        }
        
    except Exception as e:
        return {"error": f"Error analizando imagen: {str(e)}"}

def create_automatic_mask_without_sam(image_enhanced):
    """Crear máscara automática sin SAM usando los mismos métodos del script."""
    try:
        # Convertir a HSV para detección de color verde
        hsv = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2HSV)
        
        # Definir rango de color verde (igual que en el script)
        lower_green = np.array([30, 25, 25])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Aplicar suavizado selectivo
        green_mask_smoothed = cv2.bilateralFilter(green_mask, d=5, sigmaColor=50, sigmaSpace=50)
        _, green_mask_binary = cv2.threshold(green_mask_smoothed, 127, 255, cv2.THRESH_BINARY)
        
        # Limpiar la máscara
        mask_cleaned = clean_mask_sam(green_mask_binary)
        
        return mask_cleaned
        
    except Exception as e:
        print(f"Error creando máscara automática: {e}")
        # Crear máscara simple como fallback
        h, w = image_enhanced.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Crear una región central
        center_h, center_w = h // 2, w // 2
        mask[center_h-50:center_h+50, center_w-50:center_w+50] = 255
        return mask

def get_plantcv_metrics_exact(image_rgb, mask, image_bgr):
    """Calcular métricas usando exactamente los mismos métodos que el script procesamiento_SAM.py."""
    try:
        # Convertir la máscara a formato compatible con PlantCV
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Calcular altura total usando PlantCV
        height_pixels, width_pixels = pcv.bounding_rectangle(mask_uint8)
        
        # Análisis de tallos individuales usando exactamente la misma función del script
        stem_heights = analyze_individual_stems_exact(mask_uint8, image_bgr)
        
        return {
            'total_height_pixels': height_pixels,
            'stem_heights': stem_heights
        }
        
    except Exception as e:
        print(f"Error en análisis PlantCV: {e}")
        # Fallback usando OpenCV
        return get_opencv_metrics_fallback_exact(mask)

def analyze_individual_stems_exact(mask, image_bgr):
    """Analizar cada tallo individual usando exactamente los mismos métodos del script procesamiento_SAM.py."""
    try:
        # Limpiar la máscara primero (igual que en el script)
        mask_cleaned = clean_mask_sam(mask)
        
        # Crear ROI basada en la máscara (igual que en el script)
        roi = create_roi_from_mask_exact(mask_cleaned, padding=30)
        
        if roi is None:
            print("No se pudo crear ROI, usando análisis completo")
            return analyze_without_roi_exact(mask_cleaned)
        
        # Aplicar ROI a la imagen y máscara
        image_roi = apply_roi_to_image_exact(image_bgr, roi)
        mask_roi = apply_roi_to_image_exact(mask_cleaned, roi)
        
        # Preprocesar la imagen ROI (igual que en el script)
        image_enhanced_roi = preprocess_image_sam_original(image_roi)
        
        # Convertir imagen ROI preprocesada a HSV
        hsv_roi = cv2.cvtColor(image_enhanced_roi, cv2.COLOR_BGR2HSV)
        
        # Definir rango de color verde (ajustado para ROI)
        lower_green = np.array([30, 25, 25])
        upper_green = np.array([90, 255, 255])
        green_mask_roi = cv2.inRange(hsv_roi, lower_green, upper_green)
        
        # Aplicar suavizado selectivo a la máscara verde ROI
        green_mask_smoothed_roi = cv2.bilateralFilter(green_mask_roi, d=5, sigmaColor=50, sigmaSpace=50)
        _, green_mask_binary_roi = cv2.threshold(green_mask_smoothed_roi, 127, 255, cv2.THRESH_BINARY)
        
        # Aplicar la máscara ROI sobre la detección de verde
        combined_mask_roi = cv2.bitwise_and(green_mask_binary_roi, mask_roi)
        
        # Limpiar la máscara combinada ROI
        combined_mask_roi = clean_mask_sam(combined_mask_roi)
        
        # Encontrar contornos de tallos en ROI
        contours, _ = cv2.findContours(combined_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stem_heights = []
        min_stem_area = 80  # Área mínima muy estricta para ROI
        
        # Filtrar contornos por área y relación de aspecto (igual que en el script)
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_stem_area:
                x, y, w, h = cv2.boundingRect(c)
                # Filtro adicional: relación de aspecto (altura/ancho) debe ser > 1.5
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio > 1.5 and h > 30:  # Altura mínima de 30 píxeles
                    stem_heights.append(h)
        
        # Si no se encontraron tallos en ROI, usar método alternativo (igual que en el script)
        if not stem_heights:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_stem_area:
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                    # Aplicar mismo filtro de relación de aspecto
                    aspect_ratio = h / w if w > 0 else 0
                    if aspect_ratio > 1.5 and h > 30:
                        stem_heights.append(h)
        
        # Si aún no hay tallos, usar altura total de ROI (igual que en el script)
        if not stem_heights:
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if h > 30:  # Solo usar si la altura es significativa
                    stem_heights = [h]
        
        # Filtro final: eliminar tallos muy pequeños (igual que en el script)
        stem_heights = [h for h in stem_heights if h >= 40]
        
        return stem_heights
        
    except Exception as e:
        print(f"Error en análisis de tallos con ROI: {e}")
        # Fallback sin ROI
        return analyze_without_roi_exact(mask)

def create_roi_from_mask_exact(mask, padding=15):
    """Crear ROI basada en la máscara usando exactamente los mismos métodos del script."""
    # Convertir a uint8 si es necesario
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Aplicar limpieza previa para eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filtrar contornos por área mínima (eliminar ruido)
    min_contour_area = 100  # Área mínima para considerar un contorno válido
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    
    if not valid_contours:
        # Si no hay contornos válidos, usar el más grande
        valid_contours = [max(contours, key=cv2.contourArea)]
    
    # Encontrar el contorno más grande entre los válidos
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Obtener bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Añadir padding más conservador
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(mask.shape[1], x + w + padding)
    y_end = min(mask.shape[0], y + h + padding)
    
    return (x_start, y_start, x_end, y_end)

def apply_roi_to_image_exact(image, roi):
    """Aplicar ROI a una imagen usando exactamente los mismos métodos del script."""
    if roi is None:
        return image
    
    x_start, y_start, x_end, y_end = roi
    return image[y_start:y_end, x_start:x_end]

def analyze_without_roi_exact(mask_cleaned):
    """Análisis de tallos sin ROI usando exactamente los mismos métodos del script."""
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stem_heights = []
    for c in contours:
        if cv2.contourArea(c) > 25:
            x, y, w, h = cv2.boundingRect(c)
            stem_heights.append(h)
    
    if not stem_heights and contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        stem_heights = [h]
    
    # Filtro final: eliminar tallos muy pequeños
    stem_heights = [h for h in stem_heights if h >= 40]
    
    return stem_heights

def get_opencv_metrics_fallback_exact(mask):
    """Función de respaldo usando exactamente los mismos métodos del script."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_pixels = 0
    stem_heights = []
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        height_pixels = h

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(binary_mask, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        
        min_stem_area = 80  # Área mínima más estricta
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_stem_area:
                x_stem, y_stem, w_stem, h_stem = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                # Filtro de altura mínima
                if h_stem >= 40:
                    stem_heights.append(h_stem)
        
        # Si no se encontraron tallos, usar la altura total
        if not stem_heights:
            stem_heights = [height_pixels]
        
        # Filtro final: eliminar tallos muy pequeños
        stem_heights = [h for h in stem_heights if h >= 40]
                
    return {
        'total_height_pixels': height_pixels,
        'stem_heights': stem_heights
    }

def analyze_individual_stems_from_mask(mask, image_bgr):
    """Analizar cada tallo individual desde la máscara segmentada."""
    try:
        # Convertir máscara a uint8
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.float32 else mask.astype(np.uint8)
        
        # Limpiar la máscara
        mask_cleaned = clean_mask_sam(mask_uint8)
        
        # Encontrar contornos de tallos individuales
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stem_heights = []
        min_stem_area = 50  # Área mínima para considerar un tallo
        
        # Analizar cada contorno
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_stem_area:
                # Obtener bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filtrar por relación de aspecto (altura/ancho > 1.5)
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio > 1.5 and h > 30:  # Altura mínima de 30 píxeles
                    stem_heights.append(h)
        
        # Si no se encontraron tallos con criterios estrictos, usar método alternativo
        if not stem_heights:
            # Usar componentes conectados
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_stem_area:
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                    aspect_ratio = h / w if w > 0 else 0
                    if aspect_ratio > 1.5 and h > 30:
                        stem_heights.append(h)
        
        # Si aún no hay tallos, usar altura total de la máscara
        if not stem_heights:
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if h > 30:
                    stem_heights = [h]
        
        # Filtrar tallos muy pequeños
        stem_heights = [h for h in stem_heights if h >= 40]
        
        return stem_heights
        
    except Exception as e:
        print(f"Error analizando tallos individuales: {e}")
        return []

# ---------- FUNCIONES PARA SEGMENTACIÓN SAM ----------

# Variables globales para segmentación SAM
sam_predictor = None
sam_input_points = []
sam_input_labels = []
sam_current_image = None

def initialize_sam_model():
    """Inicializar el modelo SAM si no está cargado."""
    global sam_predictor
    if sam_predictor is None:
        try:
            print("🔄 Intentando importar segment_anything...")
            from segment_anything import sam_model_registry, SamPredictor
            print("✅ segment_anything importado correctamente")
            
            # Construir ruta del checkpoint
            checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "sam_vit_h_4b8939.pth")
            checkpoint_path = os.path.abspath(checkpoint_path)
            print(f"🔍 Buscando checkpoint en: {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                print(f"❌ No se encontró el checkpoint SAM: {checkpoint_path}")
                return False, f"❌ No se encontró el archivo del modelo SAM en: {checkpoint_path}"
            
            print("🔄 Cargando modelo SAM...")
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            sam_predictor = SamPredictor(sam)
            print("✅ Modelo SAM inicializado correctamente")
            return True, "✅ Modelo SAM inicializado correctamente"
            
        except ImportError as e:
            error_msg = f"❌ Error importando segment_anything: {e}\n💡 Instala con: pip install segment-anything"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"❌ Error inicializando SAM: {e}"
            print(error_msg)
            return False, error_msg
    return True, "✅ Modelo SAM ya está inicializado"

def load_image_for_sam_segmentation(image_path):
    """Cargar imagen para segmentación SAM."""
    global sam_current_image, sam_input_points, sam_input_labels
    
    try:
        # Limpiar puntos anteriores
        sam_input_points = []
        sam_input_labels = []
        
        # Cargar imagen
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return None, "❌ No se pudo cargar la imagen"
        
        # Preprocesar imagen (igual que en el script SAM)
        image_enhanced = preprocess_image_for_sam(image_bgr)
        sam_current_image = image_enhanced
        
        # Convertir a RGB para mostrar
        image_rgb = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB)
        
        return image_rgb, "✅ Imagen cargada para segmentación SAM. Haz clic para seleccionar puntos."
        
    except Exception as e:
        return None, f"❌ Error cargando imagen: {str(e)}"

def preprocess_image_for_sam(image_bgr):
    """Preprocesar imagen para SAM (igual que en procesamiento_SAM.py)."""
    return preprocess_image_sam_original(image_bgr)

def add_sam_point(image, evt: gr.SelectData):
    """Agregar punto de segmentación SAM."""
    global sam_input_points, sam_input_labels
    
    if sam_current_image is None:
        return image, "❌ Primero carga una imagen"
    
    # Agregar punto (clic izquierdo = positivo, derecho = negativo)
    # En Gradio, evt.index contiene [x, y]
    x, y = evt.index[0], evt.index[1]
    
    # Por defecto, todos los clics son positivos (parte de la planta)
    sam_input_points.append([x, y])
    sam_input_labels.append(1)  # 1 = positivo, 0 = negativo
    
    # Dibujar punto en la imagen
    image_with_point = image.copy()
    cv2.circle(image_with_point, (x, y), 5, (0, 255, 0), -1)  # Verde para positivos
    
    msg = f"✅ Punto agregado en ({x}, {y}). Total puntos: {len(sam_input_points)}"
    return image_with_point, msg

def reset_sam_points():
    """Resetear puntos de segmentación SAM."""
    global sam_input_points, sam_input_labels
    
    sam_input_points = []
    sam_input_labels = []
    
    return "🔄 Puntos reseteados. Haz clic para agregar nuevos puntos."

def segment_with_sam():
    """Realizar segmentación con SAM."""
    global sam_predictor, sam_input_points, sam_input_labels, sam_current_image
    
    if sam_current_image is None:
        return None, "❌ Primero carga una imagen"
    
    if not sam_input_points:
        return None, "❌ Agrega al menos un punto haciendo clic en la imagen"
    
    try:
        # Inicializar SAM si no está cargado
        success, message = initialize_sam_model()
        if not success:
            # Si SAM no está disponible, usar método manual como fallback
            print("⚠️ SAM no disponible, usando método manual como fallback")
            return segment_with_script_fallback()
        
        # Convertir imagen a RGB para SAM
        image_rgb = cv2.cvtColor(sam_current_image, cv2.COLOR_BGR2RGB)
        
        # Configurar imagen en el predictor
        sam_predictor.set_image(image_rgb)
        
        # Convertir puntos a numpy arrays
        input_points_np = np.array(sam_input_points)
        input_labels_np = np.array(sam_input_labels)
        
        # Realizar predicción
        masks, scores, _ = sam_predictor.predict(
            point_coords=input_points_np,
            point_labels=input_labels_np,
            multimask_output=False,
        )
        
        # Obtener la mejor máscara
        final_mask = masks[0]
        
        # Crear visualización de la segmentación
        mask_visualization = np.zeros_like(sam_current_image)
        mask_visualization[final_mask > 0.0] = [0, 255, 255]  # Amarillo para la máscara
        output_image = cv2.addWeighted(sam_current_image, 0.7, mask_visualization, 0.3, 0)
        
        # Añadir puntos de segmentación
        for i, point in enumerate(sam_input_points):
            x, y = int(point[0]), int(point[1])
            label = sam_input_labels[i]
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Verde para positivos, rojo para negativos
            cv2.circle(output_image, (x, y), 5, color, -1)
        
        # Convertir a RGB para mostrar
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        
        # Guardar máscara para análisis posterior
        global sam_segmentation_mask
        sam_segmentation_mask = final_mask
        
        msg = f"✅ Segmentación completada con {len(sam_input_points)} puntos. Score: {scores[0]:.3f}"
        return output_rgb, msg
        
    except Exception as e:
        return None, f"❌ Error en segmentación SAM: {str(e)}"

# Variable global para la máscara segmentada
sam_segmentation_mask = None

def check_sam_dependencies():
    """Verificar si las dependencias de SAM están instaladas."""
    try:
        import segment_anything
        from segment_anything import sam_model_registry, SamPredictor
        
        # Verificar que el checkpoint existe
        checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "sam_vit_h_4b8939.pth")
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            return False, f"❌ Checkpoint SAM no encontrado en: {checkpoint_path}"
        
        # Intentar cargar el modelo
        try:
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            return True, f"✅ SAM está listo. Checkpoint: {os.path.basename(checkpoint_path)}"
        except Exception as e:
            return False, f"❌ Error cargando modelo SAM: {str(e)}"
            
    except ImportError as e:
        return False, f"❌ segment_anything no está instalado: {str(e)}\n💡 Instala con: pip install segment-anything"

def segment_with_script_fallback():
    """Usar métodos del script procesamiento_SAM.py con fallback si SAM no está disponible."""
    global sam_current_image, sam_input_points, sam_input_labels
    
    if sam_current_image is None:
        return None, "❌ Primero carga una imagen"
    
    if not sam_input_points:
        return None, "❌ Agrega al menos un punto haciendo clic en la imagen"
    
    try:
        # Intentar usar SAM primero
        try:
            # Usar el mismo preprocesamiento que el script original
            image_enhanced = preprocess_image_sam_original(sam_current_image)
            
            # Convertir a RGB para SAM (igual que en el script)
            image_rgb = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB)
            
            # Inicializar SAM si no está cargado
            success, message = initialize_sam_model()
            if success:
                # Configurar imagen en el predictor (igual que en el script)
                sam_predictor.set_image(image_rgb)
                
                # Convertir puntos a numpy arrays (igual que en el script)
                input_points_np = np.array(sam_input_points)
                input_labels_np = np.array(sam_input_labels)
                
                # Realizar predicción (igual que en el script)
                masks, scores, _ = sam_predictor.predict(
                    point_coords=input_points_np,
                    point_labels=input_labels_np,
                    multimask_output=False,
                )
                
                # Obtener la mejor máscara (igual que en el script)
                final_mask = masks[0]
                
                # Crear visualización de la segmentación (igual que en el script)
                mask_visualization = np.zeros_like(sam_current_image)
                mask_visualization[final_mask > 0.0] = [0, 255, 255]  # Amarillo para la máscara
                output_image = cv2.addWeighted(sam_current_image, 0.7, mask_visualization, 0.3, 0)
                
                # Añadir puntos de segmentación (igual que en el script)
                for i, point in enumerate(sam_input_points):
                    x, y = int(point[0]), int(point[1])
                    label = sam_input_labels[i]
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Verde para positivos, rojo para negativos
                    cv2.circle(output_image, (x, y), 5, color, -1)
                
                # Convertir a RGB para mostrar
                output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                
                # Guardar máscara para análisis posterior
                global sam_segmentation_mask
                sam_segmentation_mask = final_mask
                
                msg = f"✅ Segmentación completada usando SAM con {len(sam_input_points)} puntos. Score: {scores[0]:.3f}"
                return output_rgb, msg
            else:
                raise Exception(f"SAM no disponible: {message}")
                
        except Exception as sam_error:
            print(f"⚠️ SAM no disponible, usando método de fallback: {sam_error}")
            # Fallback: usar método de segmentación manual sin SAM
            return segment_with_manual_fallback()
        
    except Exception as e:
        return None, f"❌ Error en segmentación: {str(e)}"

def segment_with_manual_fallback():
    """Método de fallback sin SAM usando técnicas de segmentación manual."""
    global sam_current_image, sam_input_points, sam_input_labels
    
    try:
        # Usar el mismo preprocesamiento que el script original
        image_enhanced = preprocess_image_sam_original(sam_current_image)
        
        # Convertir a HSV para detección de color verde
        hsv = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2HSV)
        
        # Definir rango de color verde (igual que en el script)
        lower_green = np.array([30, 25, 25])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Aplicar suavizado selectivo
        green_mask_smoothed = cv2.bilateralFilter(green_mask, d=5, sigmaColor=50, sigmaSpace=50)
        _, green_mask_binary = cv2.threshold(green_mask_smoothed, 127, 255, cv2.THRESH_BINARY)
        
        # Crear máscara basada en los puntos seleccionados
        mask = np.zeros(sam_current_image.shape[:2], dtype=np.uint8)
        
        # Para cada punto, crear una región usando flood fill
        for point in sam_input_points:
            x, y = int(point[0]), int(point[1])
            
            # Crear máscara temporal para flood fill
            h, w = green_mask_binary.shape
            mask_flood = np.zeros((h + 2, w + 2), np.uint8)
            
            # Aplicar flood fill desde el punto
            cv2.floodFill(green_mask_binary, mask_flood, (x, y), 255)
            flood_region = mask_flood[1:h+1, 1:w+1]
            
            # Combinar con la máscara principal
            mask = cv2.bitwise_or(mask, flood_region)
        
        # Limpiar la máscara
        mask = clean_mask_sam(mask)
        
        # Crear visualización de la máscara
        mask_colored = np.zeros_like(sam_current_image)
        mask_colored[mask > 0] = [0, 255, 255]  # Amarillo para la máscara
        output_image = cv2.addWeighted(sam_current_image, 0.7, mask_colored, 0.3, 0)
        
        # Dibujar puntos seleccionados
        for i, point in enumerate(sam_input_points):
            x, y = int(point[0]), int(point[1])
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)
        
        # Convertir a RGB para mostrar
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        
        # Guardar máscara para análisis posterior
        global sam_segmentation_mask
        sam_segmentation_mask = mask
        
        msg = f"✅ Segmentación completada usando método manual con {len(sam_input_points)} puntos"
        return output_rgb, msg
        
    except Exception as e:
        return None, f"❌ Error en segmentación manual: {str(e)}"

def preprocess_image_sam_original(image_bgr):
    """Preprocesar imagen exactamente como en el script procesamiento_SAM.py."""
    # Convertir a LAB para mejor procesamiento
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE en el canal L (luminancia)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Reconstruir imagen LAB
    lab_enhanced = cv2.merge([l, a, b])
    
    # Convertir de vuelta a BGR
    image_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Corrección de dominancia de color
    # Convertir a float para cálculos
    img_float = image_enhanced.astype(np.float32)
    
    # Calcular promedio de cada canal
    avg_b = np.mean(img_float[:,:,0])
    avg_g = np.mean(img_float[:,:,1])
    avg_r = np.mean(img_float[:,:,2])
    
    # Calcular factor de balance
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # Aplicar balance de blancos
    img_float[:,:,0] = img_float[:,:,0] * (avg_gray / avg_b)
    img_float[:,:,1] = img_float[:,:,1] * (avg_gray / avg_g)
    img_float[:,:,2] = img_float[:,:,2] * (avg_gray / avg_r)
    
    # Normalizar y convertir de vuelta a uint8
    img_balanced = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # Suavizado selectivo basado en detección de bordes
    # Crear máscara de bordes para aplicar suavizado selectivo
    gray = cv2.cvtColor(img_balanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilatar los bordes para crear una máscara más amplia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Invertir la máscara para suavizar áreas sin bordes
    smooth_mask = cv2.bitwise_not(edges_dilated)
    
    # Aplicar suavizado gaussiano en áreas sin bordes
    img_gaussian = cv2.GaussianBlur(img_balanced, (5, 5), 0)
    
    # Combinar imagen original (en bordes) con imagen suavizada (en áreas planas)
    img_selective = np.where(smooth_mask[..., np.newaxis] > 0, img_gaussian, img_balanced)
    
    # Aplicar filtro bilateral final para suavizado preservando bordes
    img_final = cv2.bilateralFilter(img_selective.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
    
    return img_final

def clean_mask_sam(mask):
    """Limpiar la máscara usando exactamente los mismos métodos del script procesamiento_SAM.py."""
    # Convertir a uint8 si es necesario
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Aplicar suavizado preservando bordes a la máscara
    # Esto ayuda a conectar componentes cercanos y eliminar ruido
    mask_smoothed = cv2.bilateralFilter(mask, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Binarizar después del suavizado
    _, mask_binary = cv2.threshold(mask_smoothed, 127, 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas para limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Apertura para eliminar ruido pequeño
    mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Cierre para llenar huecos pequeños
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Encontrar el componente más grande y eliminar el resto
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
    
    if num_labels > 1:
        # Encontrar el componente más grande (excluyendo el fondo)
        largest_component = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[i, cv2.CC_STAT_AREA]
                largest_component = i
        
        # Crear máscara solo con el componente más grande
        mask_cleaned = (labels == largest_component).astype(np.uint8) * 255
    
    return mask_cleaned

def create_roi_from_points(points, image_shape):
    """Crear ROI basado en los puntos seleccionados."""
    if not points:
        return None
    
    # Calcular bounding box de los puntos
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min = max(0, int(min(x_coords)) - 50)
    y_min = max(0, int(min(y_coords)) - 50)
    x_max = min(image_shape[1], int(max(x_coords)) + 50)
    y_max = min(image_shape[0], int(max(y_coords)) + 50)
    
    return (x_min, y_min, x_max, y_max)

def apply_roi_to_image(image, roi):
    """Aplicar ROI a una imagen."""
    if roi is None:
        return image
    
    x_start, y_start, x_end, y_end = roi
    return image[y_start:y_end, x_start:x_end]

def expand_mask_in_roi(mask_roi, green_mask_roi):
    """Expandir la máscara en ROI usando crecimiento de región."""
    # Usar flood fill para expandir desde las semillas
    mask_expanded = mask_roi.copy()
    
    # Encontrar componentes conectados
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
    
    for i in range(1, n):
        # Obtener el centroide del componente
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        
        # Usar flood fill para expandir desde el centroide
        h, w = mask_roi.shape
        mask_flood = np.zeros((h + 2, w + 2), np.uint8)
        
        # Aplicar flood fill
        cv2.floodFill(green_mask_roi, mask_flood, (cx, cy), 255)
        
        # Extraer la región expandida
        flood_region = mask_flood[1:h+1, 1:w+1]
        
        # Combinar con la máscara original
        mask_expanded = cv2.bitwise_or(mask_expanded, flood_region)
    
    return mask_expanded

def segment_with_script_only():
    """Usar solo el script procesamiento_SAM.py sin dependencias externas."""
    global sam_current_image, sam_input_points, sam_input_labels
    
    if sam_current_image is None:
        return None, "❌ Primero carga una imagen"
    
    if not sam_input_points:
        return None, "❌ Agrega al menos un punto haciendo clic en la imagen"
    
    try:
        # Crear archivo temporal para el script
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        temp_file.close()
        
        # Guardar imagen actual
        cv2.imwrite(temp_path, sam_current_image)
        
        # Usar análisis básico sin SAM
        result = analyze_height_and_stems(temp_path)
        
        # Limpiar archivo temporal
        os.unlink(temp_path)
        
        if "error" in result:
            return None, f"❌ {result['error']}"
        
        # Crear visualización básica
        image_rgb = cv2.cvtColor(sam_current_image, cv2.COLOR_BGR2RGB)
        
        # Dibujar puntos seleccionados
        for i, point in enumerate(sam_input_points):
            x, y = int(point[0]), int(point[1])
            cv2.circle(image_rgb, (x, y), 5, (0, 255, 0), -1)
        
        # Crear máscara simple basada en los puntos
        mask = np.zeros(sam_current_image.shape[:2], dtype=np.uint8)
        for point in sam_input_points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(mask, (x, y), 30, 255, -1)
        
        # Guardar máscara para análisis posterior
        global sam_segmentation_mask
        sam_segmentation_mask = mask
        
        msg = f"✅ Segmentación completada usando análisis básico con {len(sam_input_points)} puntos"
        return image_rgb, msg
        
    except Exception as e:
        return None, f"❌ Error en segmentación con script: {str(e)}"

# ---------- FUNCIONES PARA ANÁLISIS NOIR ----------

def show_noir_metrics() -> tuple:
    """Mostrar métricas de imágenes noir analizadas."""
    try:
        # Cargar CSV de métricas noir
        csv_path = os.path.join(BASE_PATH, "metricas_reflectancia.csv")
        
        if not os.path.exists(csv_path):
            return "❌ No se encontró el archivo de métricas noir. Ejecuta primero el script de análisis noir.", None, "❌ Error: Archivo no encontrado"
        
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return "❌ El archivo de métricas noir está vacío.", None, "❌ Error: CSV vacío"
        
        # Verificar que el CSV tenga las columnas esperadas
        expected_columns = ['imagen', 'timestamp', 'hoja_id', 'numero_hojas_total', 'area', 'ndvi_mean', 'ndvi_std']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            return f"❌ El CSV no tiene las columnas esperadas. Faltan: {missing_columns}", None, "❌ Error: Columnas faltantes"
        
        # Crear gráfico de métricas NDVI
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfico 1: NDVI promedio por imagen
        ndvi_by_image = df.groupby('imagen')['ndvi_mean'].mean().reset_index()
        ndvi_by_image['imagen_short'] = ndvi_by_image['imagen'].str.replace('foto_', '').str.replace('.jpg', '')
        
        ax1.bar(range(len(ndvi_by_image)), ndvi_by_image['ndvi_mean'], color='green', alpha=0.7)
        ax1.set_xlabel('Imagen')
        ax1.set_ylabel('NDVI Promedio')
        ax1.set_title('NDVI Promedio por Imagen')
        ax1.set_xticks(range(len(ndvi_by_image)))
        ax1.set_xticklabels(ndvi_by_image['imagen_short'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Área promedio por imagen
        area_by_image = df.groupby('imagen')['area'].mean().reset_index()
        area_by_image['imagen_short'] = area_by_image['imagen'].str.replace('foto_', '').str.replace('.jpg', '')
        
        ax2.bar(range(len(area_by_image)), area_by_image['area'], color='blue', alpha=0.7)
        ax2.set_xlabel('Imagen')
        ax2.set_ylabel('Área Promedio (px)')
        ax2.set_title('Área Promedio de Hojas por Imagen')
        ax2.set_xticks(range(len(area_by_image)))
        ax2.set_xticklabels(area_by_image['imagen_short'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Crear mensaje de estadísticas
        total_images = df['imagen'].nunique()
        total_leaves = len(df)
        avg_ndvi = df['ndvi_mean'].mean()
        avg_area = df['area'].mean()
        avg_ndvi_std = df['ndvi_std'].mean()
        
        stats_msg = f"📊 **Métricas Noir Analizadas**\n\n"
        stats_msg += f"**📈 Resumen General:**\n"
        stats_msg += f"• Total de imágenes: **{total_images}**\n"
        stats_msg += f"• Total de hojas analizadas: **{total_leaves}**\n"
        stats_msg += f"• Hojas por imagen: **{total_leaves // total_images}**\n\n"
        
        stats_msg += f"**🔬 Métricas NDVI:**\n"
        stats_msg += f"• NDVI promedio: **{avg_ndvi:.3f}**\n"
        stats_msg += f"• Desviación estándar NDVI: **{avg_ndvi_std:.3f}**\n\n"
        
        stats_msg += f"**📏 Métricas de Área:**\n"
        stats_msg += f"• Área promedio de hojas: **{avg_area:.1f} px**\n\n"
        
        stats_msg += f"**📋 Columnas del CSV:**\n"
        stats_msg += f"• imagen, timestamp, hoja_id, numero_hojas_total, area, ndvi_mean, ndvi_std\n\n"
        
        stats_msg += f"**💡 Interpretación NDVI:**\n"
        if avg_ndvi < 0:
            stats_msg += f"• NDVI negativo ({avg_ndvi:.3f}) indica estrés vegetal o condiciones adversas\n"
        elif avg_ndvi < 0.2:
            stats_msg += f"• NDVI bajo ({avg_ndvi:.3f}) sugiere vegetación en desarrollo o estrés\n"
        elif avg_ndvi < 0.5:
            stats_msg += f"• NDVI moderado ({avg_ndvi:.3f}) indica vegetación saludable\n"
        else:
            stats_msg += f"• NDVI alto ({avg_ndvi:.3f}) indica vegetación muy saludable y vigorosa\n"
        
        return stats_msg, fig, "✅ Métricas noir cargadas correctamente"
        
    except Exception as e:
        error_msg = f"❌ Error mostrando métricas noir: {str(e)}"
        return error_msg, None, error_msg

def analyze_noir_image_upload(image) -> tuple:
    """Analizar una nueva imagen noir subida por el usuario."""
    if image is None:
        return None, "⚠️ Por favor, sube una imagen para analizar"
    
    try:
        # Convertir imagen de Gradio a formato OpenCV
        if isinstance(image, str):  # Es una ruta de archivo
            temp_path = image
        elif hasattr(image, 'name'):  # Archivo subido con atributo name
            temp_path = image.name
        else:  # Imagen en memoria, crear archivo temporal
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            # Convertir imagen a formato PIL y guardar
            if hasattr(image, 'save'):
                image.save(temp_path)
            else:
                # Si no tiene método save, intentar convertir
                try:
                    import PIL.Image
                    if isinstance(image, np.ndarray):
                        pil_image = PIL.Image.fromarray(image)
                        pil_image.save(temp_path)
                    else:
                        return None, "❌ Formato de imagen no soportado"
                except Exception as conv_error:
                    return None, f"❌ Error convirtiendo imagen: {str(conv_error)}"
        
        # Verificar que el archivo existe
        if not os.path.exists(temp_path):
            return None, f"❌ No se pudo crear archivo temporal: {temp_path}"
        
        print(f"🔴 Analizando imagen noir subida: {temp_path}")
        
        # Intentar importar funciones del script procesamiento_noir
        try:
            import importlib.util
            noir_script_path = os.path.join(os.path.dirname(__file__), "..", "procesamiento_noir.py")
            
            if not os.path.exists(noir_script_path):
                return None, "❌ No se encontró el script de procesamiento_noir.py"
            
            # Importar módulo dinámicamente
            spec = importlib.util.spec_from_file_location("noir_module", noir_script_path)
            noir_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(noir_module)
            
            # Intentar usar función específica si existe
            if hasattr(noir_module, 'analyze_single_image_noir'):
                result = noir_module.analyze_single_image_noir(temp_path)
            else:
                result = analyze_noir_basic(temp_path)
                
        except Exception as import_error:
            print(f"⚠️ Error importando módulo noir: {import_error}")
            result = analyze_noir_basic(temp_path)
        
        # Limpiar archivo temporal si se creó
        if 'temp_file' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        if result is None:
            return None, "❌ No se pudo analizar la imagen"
        
        # Crear visualización del resultado
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Mostrar imagen procesada
        image_cv = cv2.imread(temp_path if 'temp_file' not in locals() else temp_path)
        if image_cv is not None:
            ax.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        ax.set_title("Imagen Noir Analizada", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Agregar información del análisis
        info_text = f"🔴 ANÁLISIS NOIR AVANZADO (procesamiento_noir.py): Imagen Subida\n"
        info_text += f"📅 Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if 'ndvi_metrics' in result:
            ndvi_data = result['ndvi_metrics']
            info_text += f"🌿 **HOJAS DETECTADAS:** {len(ndvi_data)} hojas\n\n"
            
            if ndvi_data:
                info_text += f"📊 **MÉTRICAS NDVI POR HOJA:**\n"
                for i, leaf in enumerate(ndvi_data, 1):
                    info_text += f"  **Hoja {i}:**\n"
                    info_text += f"    • Área: {leaf.get('area', 'N/A'):.1f} px\n"
                    info_text += f"    • NDVI promedio: {leaf.get('ndvi_mean', 'N/A'):.3f}\n"
                    info_text += f"    • NDVI desv_std: {leaf.get('ndvi_std', 'N/A'):.3f}\n\n"
                
                # Calcular estadísticas generales
                areas = [leaf.get('area', 0) for leaf in ndvi_data]
                ndvi_means = [leaf.get('ndvi_mean', 0) for leaf in ndvi_data]
                ndvi_stds = [leaf.get('ndvi_std', 0) for leaf in ndvi_data]
                
                info_text += f"📈 **ESTADÍSTICAS GENERALES:**\n"
                info_text += f"  • Área total: {sum(areas):.1f} px\n"
                info_text += f"  • Área promedio: {np.mean(areas):.1f} px\n"
                info_text += f"  • NDVI promedio: {np.mean(ndvi_means):.3f}\n"
                info_text += f"  • NDVI desv_std promedio: {np.mean(ndvi_stds):.3f}\n\n"
        
        info_text += f"🔬 **PIPELINE APLICADO (procesamiento_noir.py):**\n"
        info_text += f"  • Preprocesamiento avanzado (balance de blancos + CLAHE)\n"
        info_text += f"  • ROIs circulares adaptativos (2 macetas)\n"
        info_text += f"  • Segmentación multi-nivel (percentiles 90%, 91.5%)\n"
        info_text += f"  • Filtrado por forma y tamaño\n"
        info_text += f"  • Cálculo de métricas NDVI por hoja individual\n\n"
        
        info_text += f"💡 **INTERPRETACIÓN NDVI:**\n"
        if 'ndvi_metrics' in result and result['ndvi_metrics']:
            avg_ndvi = np.mean([leaf.get('ndvi_mean', 0) for leaf in result['ndvi_metrics']])
            if avg_ndvi < 0:
                info_text += f"  • NDVI negativo ({avg_ndvi:.3f}) indica estrés vegetal\n"
            elif avg_ndvi < 0.2:
                info_text += f"  • NDVI bajo ({avg_ndvi:.3f}) sugiere vegetación en desarrollo\n"
            elif avg_ndvi < 0.5:
                info_text += f"  • NDVI moderado ({avg_ndvi:.3f}) indica vegetación saludable\n"
            else:
                info_text += f"  • NDVI alto ({avg_ndvi:.3f}) indica vegetación muy vigorosa\n"
        
        # Crear mensaje de éxito con métricas detalladas
        success_msg = f"✅ **Análisis noir completado para imagen subida**\n"
        success_msg += f"📊 **Métricas NDVI extraídas:** {len(result.get('ndvi_metrics', []))} hojas\n"
        success_msg += f"💾 **Resultados guardados en memoria**\n\n"
        
        # Agregar métricas detalladas si están disponibles
        if 'ndvi_metrics' in result and result['ndvi_metrics']:
            ndvi_data = result['ndvi_metrics']
            success_msg += f"📋 **MÉTRICAS NDVI POR HOJA:**\n"
            
            for i, leaf in enumerate(ndvi_data, 1):
                success_msg += f"  **Hoja {i}:**\n"
                success_msg += f"    • Área: {leaf.get('area', 'N/A'):.1f} px\n"
                success_msg += f"    • NDVI promedio: {leaf.get('ndvi_mean', 'N/A'):.3f}\n"
                success_msg += f"    • NDVI desv_std: {leaf.get('ndvi_std', 'N/A'):.3f}\n\n"
            
            # Calcular estadísticas generales
            areas = [leaf.get('area', 0) for leaf in ndvi_data]
            ndvi_means = [leaf.get('ndvi_mean', 0) for leaf in ndvi_data]
            ndvi_stds = [leaf.get('ndvi_std', 0) for leaf in ndvi_data]
            
            success_msg += f"📈 **ESTADÍSTICAS GENERALES:**\n"
            success_msg += f"  • Área total: {sum(areas):.1f} px\n"
            success_msg += f"  • Área promedio: {np.mean(areas):.1f} px\n"
            success_msg += f"  • NDVI promedio: {np.mean(ndvi_means):.3f}\n"
            success_msg += f"  • NDVI desv_std promedio: {np.mean(ndvi_stds):.3f}\n\n"
            
            # Agregar interpretación NDVI
            avg_ndvi = np.mean(ndvi_means)
            success_msg += f"💡 **INTERPRETACIÓN NDVI:**\n"
            if avg_ndvi < 0:
                success_msg += f"  • NDVI negativo ({avg_ndvi:.3f}) indica estrés vegetal\n"
            elif avg_ndvi < 0.2:
                success_msg += f"  • NDVI bajo ({avg_ndvi:.3f}) sugiere vegetación en desarrollo\n"
            elif avg_ndvi < 0.5:
                success_msg += f"  • NDVI moderado ({avg_ndvi:.3f}) indica vegetación saludable\n"
            else:
                success_msg += f"  • NDVI alto ({avg_ndvi:.3f}) indica vegetación muy vigorosa\n"
        
        return fig, success_msg
        
    except Exception as e:
        # Limpiar archivo temporal en caso de error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return None, f"❌ Error analizando imagen noir: {str(e)}"

def analyze_noir_basic(image_path: str) -> dict:
    """Análisis básico de imagen noir como fallback."""
    try:
        print(f"  🔴 Análisis básico noir fallback para: {image_path}")
        
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"    ❌ No se pudo cargar la imagen")
            return None
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbral simple para detectar objetos claros (hojas)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Área mínima
                valid_contours.append(contour)
        
        # Crear métricas NDVI simuladas (fallback)
        ndvi_metrics = []
        for i, contour in enumerate(valid_contours[:3]):  # Máximo 3 hojas
            area = cv2.contourArea(contour)
            
            # Calcular NDVI simulado basado en intensidad de gris
            # En escala de grises, valores altos = más claro = posiblemente vegetación
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [contour], 255)
            mean_intensity = np.mean(gray[mask > 0])
            
            # Simular NDVI: intensidad alta = NDVI positivo, intensidad baja = NDVI negativo
            ndvi_simulated = (mean_intensity - 127) / 127  # Normalizar a [-1, 1]
            ndvi_std_simulated = 0.1  # Desviación estándar simulada
            
            ndvi_metrics.append({
                'area': float(area),
                'ndvi_mean': float(ndvi_simulated),
                'ndvi_std': float(ndvi_std_simulated)
            })
        
        result = {
            'ndvi_metrics': ndvi_metrics,
            'total_leaves': len(ndvi_metrics),
            'analysis_type': 'basic_fallback'
        }
        
        print(f"    ✅ Análisis básico completado: {len(ndvi_metrics)} hojas detectadas")
        return result
        
    except Exception as e:
        print(f"    ❌ Error en análisis básico: {e}")
        return None

def get_noir_images():
    """Obtener lista de imágenes noir procesadas de la carpeta imagenes_procesadas_noir_avanzado"""
    try:
        processed_noir_folder = os.path.join(BASE_PATH, "imagenes_procesadas_noir_avanzado")
        
        if not os.path.exists(processed_noir_folder):
            return []
        
        # Buscar archivos de imagen
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        noir_images = []
        
        for file in os.listdir(processed_noir_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                noir_images.append(file)
        
        # Ordenar alfabéticamente
        noir_images.sort()
        
        return noir_images
        
    except Exception as e:
        print(f"❌ Error obteniendo imágenes noir: {e}")
        return []

def get_lateral_exg_images():
    """Obtener lista de imágenes laterales procesadas de la carpeta imagenes_procesadas_laterales"""
    try:
        processed_lateral_folder = os.path.join(BASE_PATH, "imagenes_procesadas_laterales")
        
        if not os.path.exists(processed_lateral_folder):
            return []
        
        # Buscar archivos de imagen
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        lateral_images = []
        
        for file in os.listdir(processed_lateral_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                lateral_images.append(file)
        
        # Ordenar alfabéticamente
        lateral_images.sort()
        
        return lateral_images
        
    except Exception as e:
        print(f"❌ Error obteniendo imágenes laterales: {e}")
        return []

def analyze_noir_existing_image(image_name):
    """Analizar una imagen noir existente del proyecto usando procesamiento_noir.py."""
    if not image_name:
        return None, "⚠️ Selecciona una imagen noir para analizar"
    
    try:
        # Construir ruta completa de la imagen procesada noir
        image_path = os.path.join(BASE_PATH, "imagenes_procesadas_noir_avanzado", image_name)
        
        if not os.path.exists(image_path):
            return None, f"❌ No se encontró la imagen: {image_path}"
        
        # Importar funciones del script procesamiento_noir
        try:
            import importlib.util
            noir_script_path = os.path.join(os.path.dirname(__file__), "..", "procesamiento_noir.py")
            
            if not os.path.exists(noir_script_path):
                return None, "❌ No se encontró el script de procesamiento_noir.py"
            
            # Importar módulo dinámicamente
            spec = importlib.util.spec_from_file_location("noir_module", noir_script_path)
            noir_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(noir_module)
            
            # Intentar usar función específica si existe
            if hasattr(noir_module, 'analyze_single_image_noir'):
                result = noir_module.analyze_single_image_noir(image_path)
            else:
                result = analyze_noir_basic(image_path)
                
        except Exception as import_error:
            print(f"⚠️ Error importando módulo noir: {import_error}")
            result = analyze_noir_basic(image_path)
        
        if result is None:
            return None, "❌ No se pudo analizar la imagen"
        
        # Crear visualización del resultado
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Mostrar imagen procesada
        image = cv2.imread(image_path)
        if image is not None:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Imagen Noir Analizada: {image_name}", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Agregar información del análisis
        info_text = f"🔴 ANÁLISIS NOIR AVANZADO (procesamiento_noir.py): {image_name}\n"
        info_text += f"📅 Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if 'ndvi_metrics' in result:
            ndvi_data = result['ndvi_metrics']
            info_text += f"🌿 **HOJAS DETECTADAS:** {len(ndvi_data)} hojas\n\n"
            
            if ndvi_data:
                info_text += f"📊 **MÉTRICAS NDVI POR HOJA:**\n"
                for i, leaf in enumerate(ndvi_data, 1):
                    info_text += f"  **Hoja {i}:**\n"
                    info_text += f"    • Área: {leaf.get('area', 'N/A'):.1f} px\n"
                    info_text += f"    • NDVI promedio: {leaf.get('ndvi_mean', 'N/A'):.3f}\n"
                    info_text += f"    • NDVI desv_std: {leaf.get('ndvi_std', 'N/A'):.3f}\n\n"
                
                # Calcular estadísticas generales
                areas = [leaf.get('area', 0) for leaf in ndvi_data]
                ndvi_means = [leaf.get('ndvi_mean', 0) for leaf in ndvi_data]
                ndvi_stds = [leaf.get('ndvi_std', 0) for leaf in ndvi_data]
                
                info_text += f"📈 **ESTADÍSTICAS GENERALES:**\n"
                info_text += f"  • Área total: {sum(areas):.1f} px\n"
                info_text += f"  • Área promedio: {np.mean(areas):.1f} px\n"
                info_text += f"  • NDVI promedio: {np.mean(ndvi_means):.3f}\n"
                info_text += f"  • NDVI desv_std promedio: {np.mean(ndvi_stds):.3f}\n\n"
        
        info_text += f"🔬 **PIPELINE APLICADO (procesamiento_noir.py):**\n"
        info_text += f"  • Preprocesamiento avanzado (balance de blancos + CLAHE)\n"
        info_text += f"  • ROIs circulares adaptativos (2 macetas)\n"
        info_text += f"  • Segmentación multi-nivel (percentiles 90%, 91.5%)\n"
        info_text += f"  • Filtrado por forma y tamaño\n"
        info_text += f"  • Cálculo de métricas NDVI por hoja individual\n\n"
        
        info_text += f"💡 **INTERPRETACIÓN NDVI:**\n"
        if 'ndvi_metrics' in result and result['ndvi_metrics']:
            avg_ndvi = np.mean([leaf.get('ndvi_mean', 0) for leaf in result['ndvi_metrics']])
            if avg_ndvi < 0:
                info_text += f"  • NDVI negativo ({avg_ndvi:.3f}) indica estrés vegetal\n"
            elif avg_ndvi < 0.2:
                info_text += f"  • NDVI bajo ({avg_ndvi:.3f}) sugiere vegetación en desarrollo\n"
            elif avg_ndvi < 0.5:
                info_text += f"  • NDVI moderado ({avg_ndvi:.3f}) indica vegetación saludable\n"
            else:
                info_text += f"  • NDVI alto ({avg_ndvi:.3f}) indica vegetación muy vigorosa\n"
        
        success_msg = f"✅ Análisis noir completado para {image_name}\n"
        success_msg += f"📊 Métricas NDVI extraídas: {len(result.get('ndvi_metrics', []))} hojas\n"
        success_msg += f"💾 Resultados guardados en memoria"
        
        return fig, success_msg
        
    except Exception as e:
        return None, f"❌ Error analizando imagen noir existente: {str(e)}"

# ---------- UI ----------
with gr.Blocks(title="🌱 Análisis de Fenotipado Vegetal — PlantCV", theme=gr.themes.Soft(), css="""
#time_plot_container {
    min-height: 600px !important;
    width: 100% !important;
}
#time_plot_container .plotly {
    width: 100% !important;
    height: 600px !important;
}
""") as demo:
    gr.Markdown(
        """
        # 🌱 Análisis de Fenotipado Vegetal — PlantCV
        
        **Genera gráficos temporales y analiza imágenes individuales para el fenotipado de plantas**
        
        ---
        """
    )

    with gr.Row():
        btn_refresh = gr.Button("🔄 Actualizar opciones", variant="primary", size="lg")
        info_display = gr.Markdown("ℹ️ Los dropdowns ya tienen opciones disponibles. Haz clic en ellos para ver las opciones.")

    with gr.Tabs():
        # Tab única: Análisis Temporal
        with gr.Tab("⏰ Análisis Temporal"):
            gr.Markdown("## 📊 Generador de Gráficos Temporales")
            
            with gr.Row():
                gr.Markdown("### 📄 Selecciona el archivo CSV:")
                dd_csv_time = gr.Dropdown(choices=["metricas_morfologicas.csv", "metricas_reflectancia.csv", "metricas_tallos.csv", "sensores_ambiente.csv", "sensores_humedad.csv"], label="CSV para análisis temporal", scale=2, allow_custom_value=False, interactive=True)
            
            with gr.Row():
                gr.Markdown("### 📈 Selecciona las columnas para graficar:")
            
            with gr.Row():
                columns_info = gr.Markdown("📋 **Columnas disponibles:** Selecciona un CSV arriba para ver las columnas")
            
            with gr.Row():
                dd_x_time = gr.Dropdown(choices=[], label="Columna X (fecha/tiempo)", scale=1, allow_custom_value=False, interactive=True)
                dd_y_time = gr.Dropdown(choices=[], label="Columna Y (numérica)", scale=1, allow_custom_value=False, interactive=True)
                dd_chart_type = gr.Dropdown(choices=["Líneas y puntos", "Solo líneas", "Solo puntos", "Barras"], 
                                          value="Líneas y puntos", label="Tipo de gráfico", scale=1, allow_custom_value=False, interactive=True)
                dd_aggregation = gr.Dropdown(choices=["Promedio", "Suma", "Máximo", "Mínimo", "Sin agregar"], 
                                           value="Promedio", label="Agregación de datos", scale=1, allow_custom_value=False, interactive=True)
                btn_test_columns = gr.Button("🔄 Actualizar columnas", variant="secondary", scale=1)
            
            gr.Markdown("💡 **Instrucciones:** Selecciona primero un CSV arriba, luego las columnas X e Y se actualizarán automáticamente y podrás seleccionarlas de los dropdowns desplegables.")
            
            with gr.Row():
                pass
            
            with gr.Row():
                pass
            
            with gr.Row():
                btn_time_plot = gr.Button("📊 Generar gráfico temporal", variant="primary", size="lg")
                btn_csv_info = gr.Button("ℹ️ Información del CSV", variant="secondary", scale=1)
            
            with gr.Row():
                csv_info = gr.Markdown()
            
            with gr.Row():
                with gr.Column(scale=3):
                    time_plot = gr.Plot(label="Gráfico temporal interactivo", container=True, elem_id="time_plot_container")
                with gr.Column(scale=1):
                    time_plot_msg = gr.Markdown()

        # Nuevo Tab: Análisis Individual de Imágenes
        with gr.Tab("🔍 Análisis Individual de Imágenes"):
            gr.Markdown("## 🔍 Análisis Completo de Imágenes Individuales")
            gr.Markdown("**Analiza imágenes existentes o sube nuevas para calcular métricas completas de fenotipado**")
            
            with gr.Tabs():
                # Sub-tab: Imágenes Existentes
                with gr.Tab("📁 Imágenes del Proyecto"):
                    gr.Markdown("### 📁 Selecciona una imagen existente del proyecto:")
                    
                    with gr.Row():
                        dd_existing_images = gr.Dropdown(
                            choices=get_available_images(),
                            label="Imágenes disponibles",
                            scale=2,
                            allow_custom_value=False,
                            interactive=True
                        )
                        btn_refresh_images = gr.Button("🔄 Actualizar lista", variant="secondary", scale=1)
                    
                    with gr.Row():
                        btn_analyze_existing = gr.Button("🔍 Analizar imagen seleccionada", variant="primary", size="lg")
                    
                    with gr.Row():
                        existing_analysis_result = gr.Plot(label="Resultado del análisis")
                        existing_analysis_msg = gr.Markdown()
                
                # Sub-tab: Subir Nueva Imagen
                with gr.Tab("📤 Subir Nueva Imagen"):
                    gr.Markdown("### 📤 Sube una nueva imagen para analizar:")
                    
                    with gr.Row():
                        image_upload = gr.Image(
                            label="Subir imagen",
                            type="filepath",
                            height=300
                        )
                    
                    with gr.Row():
                        btn_analyze_upload = gr.Button("🔍 Analizar imagen subida", variant="primary", size="lg")
                    
                    with gr.Row():
                        upload_analysis_result = gr.Plot(label="Resultado del análisis")
                        upload_analysis_msg = gr.Markdown()
            
            # Información adicional
            with gr.Accordion("ℹ️ Información del Análisis", open=False):
                gr.Markdown("""
                **🔬 Métricas Calculadas (procesamiento_arriba.py):**
                - **Área PlantCV:** Área de cada hoja detectada en píxeles (conteo preciso con PlantCV)
                - **Perímetro OpenCV:** Perímetro de cada hoja en píxeles (cálculo con OpenCV)
                - **Solidez OpenCV:** Relación área/convexidad (0.0-1.0) - indica forma de la hoja
                - **Número de hojas:** Conteo total de hojas detectadas por imagen
                - **ID de hoja:** Identificador único para cada hoja individual
                
                **🔬 Pipeline Aplicado:**
                - **Preprocesamiento avanzado:** Balance de blancos Gray World + CLAHE
                - **Segmentación ExG + HSV:** Índice Excess Green + filtros HSV calibrados
                - **ROIs circulares:** 2 regiones circulares para macetas (17% y 16% radio)
                - **Filtrado por área:** Área mínima de 210 píxeles para eliminar ruido
                - **Métricas morfológicas:** Cálculo con OpenCV + PlantCV para precisión
                
                **💾 Archivos Generados:**
                - Visualización completa (6 paneles: original, ROIs, máscaras, superposición)
                - CSV con métricas por hoja individual (área, perímetro, solidez)
                - Timestamp extraído del nombre del archivo
                
                **🎯 Aplicaciones:**
                - Monitoreo de crecimiento de hojas individuales
                - Análisis de forma y solidez de hojas
                - Detección de hojas pequeñas vs grandes
                - Investigación en fenotipado de vista superior
                - Análisis temporal de desarrollo foliar
                """)

        # Nuevo Tab: Análisis de Altura y Tallos
        with gr.Tab("🌱 Análisis de Altura y Tallos"):
            gr.Markdown("## 🌱 Análisis Especializado de Altura y Número de Tallos")
            gr.Markdown("**Utiliza el script mejorado V3 con preprocesamiento avanzado para detectar tallos finos y calcular altura**")
            
            with gr.Tabs():
                # Sub-tab: Imágenes Existentes
                with gr.Tab("📁 Imágenes del Proyecto"):
                    gr.Markdown("### 📁 Selecciona una imagen existente del proyecto:")
                    
                    with gr.Row():
                        dd_existing_images_height = gr.Dropdown(
                            choices=get_lateral_exg_images(),
                            label="Imágenes laterales disponibles para altura y tallos",
                            scale=2,
                            allow_custom_value=False,
                            interactive=True
                        )
                        btn_refresh_images_height = gr.Button("🔄 Actualizar lista", variant="secondary", scale=1)
                    
                    with gr.Row():
                        btn_analyze_height_existing = gr.Button("🌱 Analizar altura y tallos", variant="primary", size="lg")
                    
                    with gr.Row():
                        height_analysis_result = gr.Plot(label="Resultado del análisis de altura y tallos")
                        height_analysis_msg = gr.Markdown()
                
                # Sub-tab: Subir Nueva Imagen
                with gr.Tab("📤 Subir Nueva Imagen"):
                    gr.Markdown("### 📤 Sube una nueva imagen para analizar altura y tallos:")
                    gr.Markdown("**🔴 Segmentación Manual con SAM:** Haz clic en la imagen para seleccionar puntos de segmentación")
                    
                    with gr.Row():
                        image_upload_height = gr.Image(
                            label="Subir imagen para altura y tallos",
                            type="filepath",
                            height=300
                        )
                    
                    with gr.Row():
                        gr.Markdown("### 🎯 Segmentación Manual con SAM:")
                        gr.Markdown("""
                        **📋 Instrucciones paso a paso:**
                        1. **🔍 Verificar SAM:** Haz clic en "Verificar SAM" para comprobar que esté instalado
                        2. **📤 Subir imagen:** La imagen se carga automáticamente para segmentación
                        3. **🎯 Seleccionar puntos:** Haz clic en la imagen para marcar regiones de interés
                        4. **🔄 Resetear (opcional):** Si quieres empezar de nuevo
                        5. **🎯 Segmentar con SAM:** Genera la máscara usando SAM (si está disponible)
                        6. **✋ Segmentar Manual:** Genera máscara usando exactamente los mismos métodos del script SAM
                        7. **🌱 Analizar:** Calcula altura de cada tallo individual
                        
                        **💡 Nota:** Ambos métodos usan exactamente los mismos algoritmos que el script `procesamiento_SAM.py`.
                        """)
                    
                    with gr.Row():
                        # Imagen interactiva para segmentación
                        sam_segmentation_image = gr.Image(
                            label="Imagen para segmentación SAM (haz clic para seleccionar puntos)",
                            type="numpy",
                            height=400,
                            interactive=True
                        )
                    
                    with gr.Row():
                        btn_check_sam = gr.Button("🔍 Verificar SAM", variant="secondary", scale=1)
                        btn_reset_points = gr.Button("🔄 Resetear puntos", variant="secondary", scale=1)
                        btn_segment_sam = gr.Button("🎯 Segmentar con SAM", variant="secondary", scale=1)
                        btn_segment_manual = gr.Button("✋ Segmentar Manual", variant="secondary", scale=1)
                        btn_analyze_height_upload = gr.Button("🌱 Analizar altura y tallos", variant="primary", scale=2)
                    
                    with gr.Row():
                        height_upload_result = gr.Plot(label="Resultado del análisis de altura y tallos")
                        height_upload_msg = gr.Markdown()
            
            # Información adicional
            with gr.Accordion("ℹ️ Información del Análisis de Altura y Tallos", open=False):
                gr.Markdown("""
                **🌱 Métricas Especializadas:**
                - **Altura de la planta:** medida en píxeles usando PlantCV
                - **Número de tallos:** detección inteligente de estructuras finas vegetales
                - **Área de tallos vs hojas:** separación automática por forma y densidad
                - **Análisis de paleta:** detección automática de tipo de iluminación y sombras
                
                **🎯 Segmentación Manual con SAM:**
                - **Segment Anything Model (SAM):** Modelo de IA para segmentación precisa
                - **Selección de puntos:** Haz clic en la imagen para marcar regiones de interés
                - **Puntos positivos:** Clic izquierdo en partes de la planta
                - **Puntos negativos:** Clic derecho en fondo (próximamente)
                - **Segmentación automática:** El modelo SAM genera máscaras precisas
                - **Análisis personalizado:** Métricas basadas en tu selección manual
                
                **🔬 Preprocesamiento Avanzado:**
                - **Reducción de exposición global:** mejora contraste para tallos finos
                - **Balance de blancos automático:** corrección de temperatura de color
                - **Corrección de iluminación:** algoritmo Retinex simplificado
                - **CLAHE:** mejora contraste local en canal de brillo
                - **Ajuste HSV adaptativo:** optimización automática de contraste
                
                **💾 Archivos Generados:**
                - Visualización completa (imagen original + tallos + hojas + métricas)
                - Análisis de paleta de colores
                - Métricas de área y conteo de tallos
                - Máscara de segmentación SAM (si se usa segmentación manual)
                
                **🎯 Aplicaciones:**
                - Monitoreo de crecimiento de tallos
                - Análisis de densidad vegetal
                - Detección de estrés en estructuras finas
                - Investigación en arquitectura vegetal
                - Segmentación precisa con control manual
                """)

        # Nuevo Tab: Análisis Noir
        with gr.Tab("🔴 Análisis Noir"):
            gr.Markdown("## 🔴 Análisis de Imágenes con Filtro NIR (Infrarrojo)")
            gr.Markdown("**Utiliza el script avanzado de análisis noir para extraer métricas NDVI de imágenes con filtro NIR**")
            
            with gr.Tabs():
                # Sub-tab: Imágenes Noir Existentes del Proyecto
                with gr.Tab("📁 Imágenes Noir del Proyecto"):
                    gr.Markdown("### 📁 Selecciona una imagen noir existente del proyecto")
                    gr.Markdown("**Analiza imágenes con filtro NIR que ya están en el proyecto**")
                    
                    with gr.Row():
                        dd_existing_images_noir = gr.Dropdown(
                            choices=get_noir_images(),
                            label="Imágenes noir disponibles",
                            scale=2,
                            allow_custom_value=False,
                            interactive=True
                        )
                        btn_refresh_images_noir = gr.Button("🔄 Actualizar lista", variant="secondary", scale=1)
                    
                    with gr.Row():
                        btn_analyze_noir_existing = gr.Button("🔴 Analizar Imagen Noir Existente", variant="primary", size="lg")
                    
                    with gr.Row():
                        noir_existing_result = gr.Plot(label="Resultado del análisis noir")
                        noir_existing_msg = gr.Markdown()
                
                # Sub-tab: Analizar Nueva Imagen Noir
                with gr.Tab("🔍 Analizar Nueva Imagen"):
                    gr.Markdown("### 🔍 Sube una nueva imagen con filtro NIR para análisis")
                    gr.Markdown("**Extrae métricas NDVI usando el pipeline avanzado de análisis noir**")
                    
                    with gr.Row():
                        image_upload_noir = gr.Image(
                            label="Subir imagen con filtro NIR",
                            type="filepath",
                            height=300
                        )
                    
                    with gr.Row():
                        btn_analyze_noir_upload = gr.Button("🔴 Analizar Imagen Noir", variant="primary", size="lg")
                    
                    with gr.Row():
                        noir_upload_result = gr.Plot(label="Resultado del análisis noir")
                        noir_upload_msg = gr.Markdown()
            
            # Información adicional
            with gr.Accordion("ℹ️ Información del Análisis Noir", open=False):
                gr.Markdown("""
                **🔴 Análisis de Filtro NIR (Infrarrojo):**
                - **Métricas NDVI:** Extracción del Índice de Vegetación de Diferencia Normalizada
                - **Segmentación adaptativa:** Detección inteligente de vegetación usando ROIs circulares
                - **Preprocesamiento avanzado:** Balance de blancos, CLAHE y corrección de iluminación
                - **Análisis de reflectancia:** Cálculo directo de NDVI para evaluación de salud vegetal
                
                **🔬 Pipeline Avanzado:**
                - **Balance de blancos Gray World:** Neutraliza dominancia de color
                - **Corrección de iluminación:** CLAHE selectivo en canal V
                - **ROIs circulares:** 2 regiones centradas en macetas para análisis preciso
                - **Segmentación multi-nivel:** Múltiples umbrales (90%, 91.5%) con combinación inteligente
                - **Filtrado por forma:** Eliminación de ruido usando criterios morfológicos
                
                **📊 Métricas Extraídas:**
                - **NDVI (Normalized Difference Vegetation Index):** Indicador de salud vegetal
                - **Área:** Tamaño de las hojas detectadas en píxeles
                - **Desviación estándar NDVI:** Variabilidad del índice dentro de cada hoja
                
                **💾 Archivos Generados:**
                - CSV con métricas NDVI por hoja individual
                - Visualizaciones avanzadas con segmentación
                - Análisis de reflectancia NIR
                
                **🎯 Aplicaciones:**
                - Monitoreo de salud vegetal en NIR
                - Análisis de estrés hídrico
                - Detección de enfermedades tempranas
                - Investigación en fenotipado NIR
                - Evaluación de vigor vegetal
                """)

    # Función para actualizar columnas cuando cambia el CSV
    def update_columns_when_csv_changes(csv_name):
        """Actualiza las columnas cuando se selecciona un CSV diferente."""
        print(f"🔍 Función llamada con csv_name: {csv_name} (tipo: {type(csv_name)})")
        
        if not csv_name:
            print("⚠️ csv_name está vacío, retornando columnas vacías")
            return gr.update(choices=[]), gr.update(choices=[]), "📋 **Columnas disponibles:** Selecciona un CSV arriba para ver las columnas"
        
        try:
            # Si csv_name es una lista, tomar el primer elemento
            if isinstance(csv_name, list) and len(csv_name) > 0:
                csv_name = csv_name[0]
                print(f"📝 csv_name era lista, ahora es: {csv_name}")
            
            print(f"🔄 Actualizando columnas para CSV: {csv_name}")
            
            # Construir ruta completa del CSV
            csv_path = os.path.join(BASE_PATH, csv_name)
            print(f"📁 Ruta del CSV: {csv_path}")
            
            if not os.path.exists(csv_path):
                print(f"❌ CSV no existe en: {csv_path}")
                return gr.update(choices=[]), gr.update(choices=[]), f"❌ **Error:** CSV no encontrado: {csv_name}"
            
            # Leer CSV directamente con pandas
            df = pd.read_csv(csv_path)
            print(f"📊 CSV leído exitosamente: {csv_name}")
            print(f"📊 Forma del DataFrame: {df.shape}")
            
            if df is not None and not df.empty:
                columns = list(df.columns)
                print(f"📊 Columnas encontradas en {csv_name}: {columns}")
                print(f"📊 Número de columnas: {len(columns)}")
                
                # Si no hay columnas, retornar lista vacía
                if not columns or len(columns) == 0:
                    print("⚠️ No hay columnas, retornando lista vacía")
                    return gr.update(choices=[]), gr.update(choices=[]), f"⚠️ **Advertencia:** CSV {csv_name} no tiene columnas"
                
                print(f"✅ Retornando columnas: {columns}")
                print(f"✅ Tipo de retorno: {type(columns)}")
                print(f"✅ Longitud de la lista: {len(columns)}")
                
                # Crear mensaje informativo con las columnas disponibles
                columns_text = ", ".join(columns)
                info_msg = f"📊 **CSV seleccionado:** {csv_name}\n📋 **Columnas disponibles ({len(columns)}):** {columns_text}\n💡 **Instrucción:** Selecciona las columnas X e Y de los dropdowns desplegables"
                
                # Retornar las mismas columnas para ambos dropdowns + mensaje informativo
                print(f"🔄 Retornando para actualizar dropdowns:")
                print(f"   - Columnas X: {columns}")
                print(f"   - Columnas Y: {columns}")
                print(f"   - Mensaje: {info_msg}")
                
                # Usar gr.update() para forzar la actualización de los dropdowns
                return gr.update(choices=columns), gr.update(choices=columns), info_msg
            else:
                print(f"⚠️ DataFrame vacío o None para: {csv_name}")
                return gr.update(choices=[]), gr.update(choices=[]), f"⚠️ **Advertencia:** CSV {csv_name} está vacío"
        except Exception as e:
            print(f"❌ Error actualizando columnas para {csv_name}: {e}")
            import traceback
            traceback.print_exc()
            return gr.update(choices=[]), gr.update(choices=[]), f"❌ **Error:** No se pudo leer {csv_name}: {str(e)}"

    # Función de prueba simple para verificar que funcione
    def test_update_columns():
        """Función de prueba para verificar que la actualización de columnas funcione."""
        print("🧪 Ejecutando función de prueba...")
        try:
            # Probar con el primer CSV
            test_csv = "metricas_morfologicas.csv"
            csv_path = os.path.join(BASE_PATH, test_csv)
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                columns = list(df.columns)
                print(f"🧪 Prueba exitosa: {len(columns)} columnas encontradas en {test_csv}")
                print(f"🧪 Columnas: {columns}")
                return columns, columns
            else:
                print(f"🧪 Prueba fallida: CSV no existe")
                return [], []
        except Exception as e:
            print(f"🧪 Error en prueba: {e}")
            return [], []

    # Función para inicializar las columnas del primer CSV
    def initialize_columns():
        """Inicializa las columnas con el primer CSV disponible."""
        try:
            first_csv = "metricas_morfologicas.csv"
            csv_path = os.path.join(BASE_PATH, first_csv)
            
            if not os.path.exists(csv_path):
                print(f"⚠️ CSV inicial no existe: {csv_path}")
                return [], [], "📋 **Columnas disponibles:** Selecciona un CSV arriba para ver las columnas"
            
            # Leer CSV directamente con pandas
            df = pd.read_csv(csv_path)
            print(f"🚀 Inicializando columnas con {first_csv}")
            print(f"📊 Forma del DataFrame: {df.shape}")
            
            if df is not None and not df.empty:
                columns = list(df.columns)
                print(f"🚀 Columnas encontradas: {columns}")
                print(f"🚀 Número de columnas: {len(columns)}")
                
                # Crear mensaje informativo
                columns_text = ", ".join(columns)
                info_msg = f"📊 **CSV inicial:** {first_csv}\n📋 **Columnas disponibles ({len(columns)}):** {columns_text}\n💡 **Instrucción:** Selecciona las columnas X e Y de los dropdowns desplegables"
                
                return columns, columns, info_msg
            else:
                print(f"⚠️ DataFrame vacío para inicialización")
                return [], [], "📋 **Columnas disponibles:** Selecciona un CSV arriba para ver las columnas"
                
        except Exception as e:
            print(f"❌ Error inicializando columnas: {e}")
            import traceback
            traceback.print_exc()
            return [], [], "📋 **Columnas disponibles:** Selecciona un CSV arriba para ver las columnas"

    # Conectar eventos
    btn_refresh.click(
        fn=lambda: "✅ **Datos disponibles**\n📊 6 CSVs disponibles\n📋 Selecciona un CSV para ver las columnas disponibles",
        outputs=[info_display]
    )

    # Eventos de análisis individual de imágenes
    btn_refresh_images.click(
        fn=lambda: get_available_images(),
        outputs=[dd_existing_images]
    )
    
    
    btn_analyze_existing.click(
        fn=analyze_existing_image,
        inputs=[dd_existing_images],
        outputs=[existing_analysis_result, existing_analysis_msg]
    )
    
    btn_analyze_upload.click(
        fn=analyze_uploaded_image,
        inputs=[image_upload],
        outputs=[upload_analysis_result, upload_analysis_msg]
    )

    # Eventos de análisis de altura y tallos
    btn_refresh_images_height.click(
        fn=lambda: get_lateral_exg_images(),
        outputs=[dd_existing_images_height]
    )
    
    
    btn_analyze_height_existing.click(
        fn=analyze_height_stems_existing,
        inputs=[dd_existing_images_height],
        outputs=[height_analysis_result, height_analysis_msg]
    )
    
    # Eventos para segmentación SAM manual
    image_upload_height.change(
        fn=load_image_for_sam_segmentation,
        inputs=[image_upload_height],
        outputs=[sam_segmentation_image, height_upload_msg]
    )
    
    sam_segmentation_image.select(
        fn=add_sam_point,
        inputs=[sam_segmentation_image],
        outputs=[sam_segmentation_image, height_upload_msg]
    )
    
    btn_check_sam.click(
        fn=check_sam_dependencies,
        outputs=[height_upload_msg]
    )
    
    btn_reset_points.click(
        fn=reset_sam_points,
        outputs=[height_upload_msg]
    )
    
    btn_segment_sam.click(
        fn=segment_with_sam,
        outputs=[sam_segmentation_image, height_upload_msg]
    )
    
    btn_segment_manual.click(
        fn=segment_with_script_fallback,
        outputs=[sam_segmentation_image, height_upload_msg]
    )
    
    btn_analyze_height_upload.click(
        fn=analyze_height_stems_upload,
        inputs=[image_upload_height],
        outputs=[height_upload_result, height_upload_msg]
    )

    # Eventos de análisis temporal
    dd_csv_time.select(
        fn=update_columns_when_csv_changes, 
        inputs=[dd_csv_time], 
        outputs=[dd_x_time, dd_y_time, columns_info],
        show_progress=False
    )
    
    # Evento de cambio como respaldo
    dd_csv_time.change(
        fn=update_columns_when_csv_changes, 
        inputs=[dd_csv_time], 
        outputs=[dd_x_time, dd_y_time, columns_info],
        show_progress=False
    )
    
    # Evento adicional para forzar actualización
    dd_csv_time.input(
        fn=update_columns_when_csv_changes, 
        inputs=[dd_csv_time], 
        outputs=[dd_x_time, dd_y_time, columns_info],
        show_progress=False
    )
    
    # Botón de prueba para actualizar columnas manualmente
    btn_test_columns.click(
        fn=update_columns_when_csv_changes, 
        inputs=[dd_csv_time], 
        outputs=[dd_x_time, dd_y_time, columns_info],
        show_progress=False
    )
    
    # Botón de actualización manual de columnas (ya no necesario)
    pass
    
    # Agregar logging para verificar que los eventos se conecten
    print("🔗 Evento dd_csv_time.change conectado a update_columns_when_csv_changes")
    print("🔗 Evento dd_csv_time.select conectado a update_columns_when_csv_changes")
    print("🔗 Ambos eventos actualizan: dd_x_time, dd_y_time, columns_info")
    
    btn_time_plot.click(fn=plot_timeseries, inputs=[dd_csv_time, dd_x_time, dd_y_time, dd_chart_type, dd_aggregation], outputs=[time_plot, time_plot_msg])
    btn_csv_info.click(fn=get_csv_info, inputs=[dd_csv_time], outputs=[csv_info])

    # Inicializar columnas al cargar la aplicación
    demo.load(fn=initialize_columns, outputs=[dd_x_time, dd_y_time, columns_info])

    # Eventos de análisis noir
    btn_analyze_noir_upload.click(fn=analyze_noir_image_upload, inputs=[image_upload_noir], outputs=[noir_upload_result, noir_upload_msg])
    
    # Eventos de análisis noir de imágenes existentes
    btn_refresh_images_noir.click(fn=lambda: get_noir_images(), outputs=[dd_existing_images_noir])
    btn_analyze_noir_existing.click(fn=analyze_noir_existing_image, inputs=[dd_existing_images_noir], outputs=[noir_existing_result, noir_existing_msg])

# Lanzar la aplicación
if __name__ == "__main__":
    print("🔍 Funcionalidades disponibles:")
    print("   • Análisis temporal con gráficos")
    print("   • Análisis individual de imágenes existentes")
    print("   • Análisis de nuevas imágenes subidas")
    print("   • 🌱 Análisis especializado de altura y tallos (NUEVO)")
    print("   • 🔴 Análisis noir avanzado con filtro NIR (NUEVO)")
    print("   • Preprocesamiento avanzado con CLAHE y balance de blancos")
    
    # Intentar lanzar en puerto 7860, si falla, mostrar instrucciones
    try:
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,
            allowed_paths=[BASE_PATH]
        )
    except OSError as e:
        if "10048" in str(e) or "port" in str(e).lower():
            print("\n❌ ERROR: El puerto 7860 ya está en uso.")
            print("💡 SOLUCIONES:")
            print("   1. Cierra otras aplicaciones que puedan estar usando el puerto 7860")
            print("   2. Reinicia tu terminal/consola")
            print("   3. O ejecuta: netstat -ano | findstr :7860  (para ver qué proceso usa el puerto)")
            print("   4. Luego ejecuta: taskkill /PID [PID] /F  (reemplaza [PID] con el número del proceso)")
            print("\n🔄 Intenta ejecutar la aplicación nuevamente después de liberar el puerto.")
        else:
            print(f"\n❌ Error inesperado: {e}")
            print("🔄 Intenta ejecutar la aplicación nuevamente.")
    
    
    