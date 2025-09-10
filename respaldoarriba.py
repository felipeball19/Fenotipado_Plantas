import cv2 #opencv segmentacion, mejroas, contornos
import numpy as np #calculos mate y arreglos
import pandas as pd #organiza datos y exporta csv
import matplotlib.pyplot as plt #visualiza imagenes
import os #crear carpetas e interactuar con archivos
import re #extraccion timestamp
from typing import Dict, List, Tuple, Any #legibilidad especifica datos salida (str)
import warnings #advertencias, mantiene salida limpia

# PlantCV para conteo de píxeles
try:
    from plantcv import plantcv as pcv
    PLANT_CV_AVAILABLE = True
    print("PlantCV disponible - usando para conteo de píxeles")
except ImportError:
    PLANT_CV_AVAILABLE = False
    print("PlantCV no disponible - usando solo OpenCV")

# Suprimir warnings
warnings.filterwarnings('ignore')

# Constantes del sistema
INPUT_DIR = "img_web_arriba"
OUTPUT_DIR = "output"
PROCESSED_DIR = "imagenes_procesadas"
CSV_PATH = os.path.join(OUTPUT_DIR, "metricas_morfologicas.csv")

# Parámetros de segmentación para hojas verdes
MIN_LEAF_AREA = 210          # Área mínima para considerar una hoja (reducido para detectar hojas pequeñas)
KERNEL_SIZE = 4              # Tamaño del kernel para morfología (ajustado)

# Rangos HSV para detección de verde (equilibrio entre selectivo y sensible)
H_MIN_GREEN = 32            # Tono mínimo para verde (ligeramente menos restrictivo)
H_MAX_GREEN = 88            # Tono máximo para verde (ligeramente menos restrictivo)
S_MIN_GREEN = 35            # Saturación mínima para verde (ligeramente menos restrictivo)
V_MIN_GREEN = 35            # Brillo mínimo para verde (ligeramente menos restrictivo)

def ensure_dirs():
    """Crear directorios de salida necesarios."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, PROCESSED_DIR), exist_ok=True)

def list_images(folder: str) -> List[str]:
    """Listar imágenes en la carpeta especificada."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = []
    
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(valid_extensions):
                images.append(os.path.join(folder, file))
    
    return sorted(images)

def extract_timestamp(filename: str) -> str:
    """Extraer timestamp del nombre del archivo."""
    # Patrón más flexible para nombres como "webcam_2025-08-27_09-00-18"
    timestamp_regex = r"(\d{4})[_\-\s]?(\d{2})[_\-\s]?(\d{2})[_\-\s]?(\d{2})[_\-\s]?(\d{2})[_\-\s]?(\d{2})"
    match = re.search(timestamp_regex, filename)
    
    if match:
        y, M, d, h, m, s = map(int, match.groups())
        return f"{y:04d}-{M:02d}-{d:02d} {h:02d}:{m:02d}:{s:02d}"
    
    # Si no coincide, intentar con patrón más simple
    simple_regex = r"(\d{4})[_\-\s](\d{2})[_\-\s](\d{2})"
    simple_match = re.search(simple_regex, filename)
    
    if simple_match:
        y, M, d = map(int, simple_match.groups())
        return f"{y:04d}-{M:02d}-{d:02d} 00:00:00"
    
    return "Sin timestamp"

def apply_advanced_preprocessing(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Preprocesamiento avanzado para imágenes de vista superior:
    - Balance de blancos Gray World
    - Corrección de iluminación con CLAHE
    - Suavizado selectivo para preservar detalles de hojas
    - Detección de paleta de colores con coordenadas del recuadro
    """
    try:
        h, w = bgr.shape[:2]
        
        # 1. BALANCE DE BLANCOS GRAY WORLD
        b, g, r = cv2.split(bgr.astype(np.float32))
        mb, mg, mr = float(b.mean()) + 1e-6, float(g.mean()) + 1e-6, float(r.mean()) + 1e-6
        k = (mb + mg + mr) / 3.0
        
        b *= k/mb
        g *= k/mg
        r *= k/mr
        
        bgr_balanced = np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
        
        # 2. CORRECCIÓN DE ILUMINACIÓN CON CLAHE
        hsv = cv2.cvtColor(bgr_balanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # CLAHE solo en el canal V (brillo) para preservar colores
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        
        # Reconstruir HSV con canal V mejorado
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 3. SUAVIZADO SELECTIVO (bilateral filter para preservar bordes)
        bgr_smooth = cv2.bilateralFilter(bgr_enhanced, 9, 75, 75)
        
        # 4. DETECCIÓN DE PALETA DE COLORES PARA CALIBRACIÓN
        # Buscar paleta de colores en la parte inferior central de la imagen
        
        roi_x1, roi_y1 = int(w * 0.05), int(h * 0.25)  # Más a la derecha (30%) y más arriba (75%)
        roi_x2, roi_y2 = int(w * 0.97), int(h * 0.98)  # Más a la izquierda (70%) y más arriba (95%)
        
        # Verificar que las coordenadas del ROI sean válidas
        roi_x1 = max(0, min(roi_x1, w-1))
        roi_y1 = max(0, min(roi_y1, h-1))
        roi_x2 = max(roi_x1+1, min(roi_x2, w))
        roi_y2 = max(roi_y1+1, min(roi_y2, h))
        
        # Extraer la región de la paleta
        roi_carta = bgr_smooth[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Coordenadas del recuadro de la paleta para visualización
        paleta_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
        
        return bgr_smooth, roi_carta, paleta_coords
        
    except Exception as e:
        print(f"   Error en preprocesamiento avanzado: {e}")
        # Fallback: imagen original con ROI central inferior
        h, w = bgr.shape[:2]
        roi_x1, roi_y1 = int(w * 0.30), int(h * 0.75)
        roi_x2, roi_y2 = int(w * 0.70), int(h * 0.95)
        return bgr, bgr[roi_y1:roi_y2, roi_x1:roi_x2], (roi_x1, roi_y1, roi_x2, roi_y2)
def define_plant_roi_arriba(bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int, int, int, int, int]]:
    """
    Define dos ROIs circulares para plantas en vista superior:
    - ROI superior para la primera maceta (más abajo y a la derecha)
    - ROI inferior para la segunda maceta (más arriba y a la derecha)
    """
    h, w = bgr.shape[:2]
    
    # Parámetros para dos ROIs circulares separados - AMBOS REDUCIDOS LIGERAMENTE
    # ROI superior (primera maceta) - más abajo y ligeramente más pequeño
    center_x1 = int(w * 0.59)  # 59% del ancho (mantener posición horizontal)
    center_y1 = int(h * 0.22)  # 22% desde arriba (mantener posición vertical)
    radius1 = int(min(w, h) * 0.17)  # 17% del lado menor (REDUCIDO de 20% a 17%)
    
    # ROI inferior (segunda maceta) - ligeramente más pequeño
    center_x2 = int(w * 0.59)  # 59% del ancho (mantener posición horizontal)
    center_y2 = int(h * 0.45)  # 45% desde arriba (mantener posición vertical)
    radius2 = int(min(w, h) * 0.16)  # 16% del lado menor (REDUCIDO de 18% a 16%)
    
    # Asegurar que los radios sean válidos usando valores escalares
    max_radius1 = min(center_x1, center_y1, w - center_x1, h - center_y1)
    max_radius2 = min(center_x2, center_y2, w - center_x2, h - center_y2)
    
    radius1 = min(radius1, max_radius1)
    radius2 = min(radius2, max_radius2)
    
    # Crear máscara combinada de ambos ROIs
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Crear coordenadas de la circunferencia
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Máscara circular superior: (x - center_x1)² + (y - center_y1)² ≤ radius1²
    mask_circle1 = (x_coords - center_x1)**2 + (y_coords - center_y1)**2 <= radius1**2
    
    # Máscara circular inferior: (x - center_x2)² + (y - center_y2)² ≤ radius2²
    mask_circle2 = (x_coords - center_x2)**2 + (y_coords - center_y2)**2 <= radius2**2
    
    # Combinar ambas máscaras
    roi_mask = np.logical_or(mask_circle1, mask_circle2).astype(np.uint8) * 255
    
    # Coordenadas para visualización (rectángulos que contienen los círculos)
    roi_x1_1 = center_x1 - radius1
    roi_y1_1 = center_y1 - radius1
    roi_x2_1 = center_x1 + radius1
    roi_y2_1 = center_y1 + radius1
    
    roi_x1_2 = center_x2 - radius2
    roi_y1_2 = center_y2 - radius2
    roi_x2_2 = center_x2 + radius2
    roi_y2_2 = center_y2 + radius2
    
    # Asegurar que las coordenadas sean válidas
    roi_x1_1 = max(0, roi_x1_1)
    roi_y1_1 = max(0, roi_y1_1)
    roi_x2_1 = min(w, roi_x2_1)
    roi_y2_1 = min(h, roi_y2_1)
    
    roi_x1_2 = max(0, roi_x1_2)
    roi_y1_2 = max(0, roi_y1_2)
    roi_x2_2 = min(w, roi_x2_2)
    roi_y2_2 = min(h, roi_y2_2)
    
    return roi_mask, (roi_x1_1, roi_y1_1, roi_x2_1, roi_y2_1, center_x1, center_y1, radius1,
                     roi_x1_2, roi_y1_2, roi_x2_2, roi_y2_2, center_x2, center_y2, radius2)

def calibrate_green_detection(bgr: np.ndarray, roi_carta: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """
    Calibra la detección de verde usando la paleta de colores y análisis de la imagen.
    Mejora la detección con rangos equilibrados para detectar hojas reales sin falsos positivos.
    """
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Análisis global de la imagen
        h_mean = float(np.mean(h))
        s_mean = float(np.mean(s))
        v_mean = float(np.mean(v))
        
        # Calcular rangos adaptativos basados en la imagen - EQUILIBRIO
        h_std = float(np.std(h))
        s_std = float(np.std(s))
        v_std = float(np.std(v))
        
        # Rangos base para verde (equilibrio entre selectivo y sensible)
        h_min = max(32, int(H_MIN_GREEN - h_std * 0.5))  # Equilibrio
        h_max = min(88, int(H_MAX_GREEN + h_std * 0.5))  # Equilibrio
        s_min = max(35, int(S_MIN_GREEN - s_std * 0.3))  # Equilibrio
        v_min = max(35, int(V_MIN_GREEN - v_std * 0.3))  # Equilibrio
        
        # Ajustar basado en la paleta de colores si está disponible
        if roi_carta is not None and roi_carta.size > 0:
            try:
                hsv_carta = cv2.cvtColor(roi_carta, cv2.COLOR_BGR2HSV)
                h_carta, s_carta, v_carta = cv2.split(hsv_carta)
                
                # Analizar todos los colores de la paleta
                h_carta_values = h_carta.flatten()
                s_carta_values = s_carta.flatten()
                v_carta_values = v_carta.flatten()
                
                # Encontrar el verde más representativo en la paleta - EQUILIBRIO
                green_mask = (h_carta_values >= 32) & (h_carta_values <= 88) & (s_carta_values >= 35) & (v_carta_values >= 35)
                if np.any(green_mask):
                    h_green = h_carta_values[green_mask]
                    s_green = s_carta_values[green_mask]
                    v_green = v_carta_values[green_mask]
                    
                    # Usar percentiles equilibrados para rangos robustos
                    h_min = max(32, int(float(np.percentile(h_green, 8)) - 8))   # Equilibrio
                    h_max = min(88, int(float(np.percentile(h_green, 92)) + 8))  # Equilibrio
                    s_min = max(35, int(float(np.percentile(s_green, 8)) - 8))  # Equilibrio
                    v_min = max(35, int(float(np.percentile(v_green, 8)) - 8))  # Equilibrio
                    
                    print(f"    Calibración equilibrada con paleta verde: H={h_min}-{h_max}, S={s_min}-255, V={v_min}-255")
                else:
                    print(f"    No se detectó verde válido en la paleta, usando rangos equilibrados por defecto")
            except Exception as e:
                print(f"     Error en calibración con paleta: {e}")
                print(f"     Continuando con rangos por defecto")
        
        # Asegurar rangos mínimos para detección robusta pero equilibrada
        h_min = max(32, h_min)  # Mínimo absoluto para verde (equilibrio)
        h_max = min(88, h_max)  # Máximo absoluto para verde (equilibrio)
        s_min = max(35, s_min)  # Mínimo absoluto para saturación (equilibrio)
        v_min = max(35, v_min)  # Mínimo absoluto para brillo (equilibrio)
        
        return {
            'h': (h_min, h_max),
            's': (s_min, 255),
            'v': (v_min, 255)
        }
        
    except Exception as e:
        print(f"     Error en calibración principal: {e}")
        # Valores por defecto seguros
        return {
            'h': (32, 88),
            's': (35, 255),
            'v': (35, 255)
        }

def enhanced_green_segmentation(bgr: np.ndarray, roi_mask: np.ndarray, 
                              hsv_ranges: Dict[str, Tuple[int, int]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Segmentación mejorada de hojas verdes usando múltiples métodos:
    - Índice ExG (Excess Green) con umbral equilibrado
    - Filtros HSV calibrados equilibrados
    - Clasificación por color y forma equilibrada
    """
    try:
        h, w = bgr.shape[:2]
        
        # 1. ÍNDICE EXG (2G-R-B) - EQUILIBRIO
        B, G, R = cv2.split(bgr)
        exg = (2 * G.astype(np.int32) - R.astype(np.int32) - B.astype(np.int32))
        exg = np.clip(exg, 0, 255).astype(np.uint8)
        
        # Umbral adaptativo para ExG (equilibrio entre selectivo y sensible)
        exg_thresh = max(5, int(float(np.percentile(exg, 96.5))))  
        mask_exg = cv2.threshold(exg, exg_thresh, 255, cv2.THRESH_BINARY)[1]
        
        # 2. FILTROS HSV CALIBRADOS EQUILIBRADOS
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, 
                              (hsv_ranges['h'][0], hsv_ranges['s'][0], hsv_ranges['v'][0]),
                              (hsv_ranges['h'][1], hsv_ranges['s'][1], hsv_ranges['v'][1]))
        
        # 3. COMBINACIÓN DE MÁSCARAS (UNIÓN para capturar más hojas reales)
        # Unión de ExG y HSV para capturar hojas que cumplan al menos una condición
        mask_combined = cv2.bitwise_or(mask_exg, mask_hsv)
        
        # Aplicar ROI
        mask_roi = cv2.bitwise_and(mask_combined, roi_mask)
        
        # 4. POSTPROCESAMIENTO MORFOLÓGICO EQUILIBRADO
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        
        # Apertura equilibrada para eliminar ruido sin perder hojas
        mask_clean = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Cierre para conectar regiones rotas
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Dilatación equilibrada para conectar componentes cercanos
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
        
        # 5. FILTRADO POR TAMAÑO EQUILIBRADO
        mask_filtered = clean_components_by_size(mask_clean, MIN_LEAF_AREA)
        
        # 6. ANÁLISIS DE RESULTADOS
        analysis_results = {
            'exg_threshold': exg_thresh,
            'hsv_ranges': hsv_ranges,
            'min_leaf_area': MIN_LEAF_AREA,
            'total_pixels': int(np.count_nonzero(mask_filtered)),
            'percentage': float(np.count_nonzero(mask_filtered)) / float(mask_filtered.size) * 100.0,
            'exg_mean': float(np.mean(exg)),
            'exg_std': float(np.std(exg))
        }
        
        return mask_filtered, analysis_results
        
    except Exception as e:
        print(f"   Error en segmentación mejorada: {e}")
        empty_mask = np.zeros_like(bgr[:,:,0])
        empty_analysis = {
            'exg_threshold': 0,
            'hsv_ranges': {'h': (0, 0), 's': (0, 0), 'v': (0, 0)},
            'min_leaf_area': MIN_LEAF_AREA,
            'total_pixels': 0,
            'percentage': 0.0,
            'exg_mean': 0.0,
            'exg_std': 0.0
        }
        return empty_mask, empty_analysis

def clean_components_by_size(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Limpieza de componentes por tamaño usando contornos."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    mask_clean = np.zeros_like(mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.fillPoly(mask_clean, [contour], 255)
    
    return mask_clean



def calculate_morphological_metrics(mask_leaves: np.ndarray) -> List[Dict[str, Any]]:
    """
    Calcula métricas morfológicas esenciales usando OpenCV + PlantCV:
    - Área, Perímetro, Solidez (comparación entre ambas librerías)
    """
    contours, _ = cv2.findContours(mask_leaves, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metrics_list = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_LEAF_AREA:
            continue
        
        # Métricas con OpenCV
        perimeter_opencv = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity_opencv = area / (hull_area + 1e-6) if hull_area > 0 else 0
        
        # Usar PlantCV solo para conteo de píxeles
        if PLANT_CV_AVAILABLE:
            try:
                # Crear máscara para esta hoja específica
                leaf_mask = np.zeros_like(mask_leaves)
                cv2.fillPoly(leaf_mask, [contour], 255)
                
                # Contar píxeles con PlantCV
                pixel_count = np.count_nonzero(leaf_mask)
                area_plantcv = float(pixel_count)
                
            except Exception as e:
                print(f"     Error en PlantCV: {e}")
                # Fallback a OpenCV
                area_plantcv = float(area)
        else:
            # Si PlantCV no está disponible, usar área de OpenCV
            area_plantcv = float(area)
           
        
        metrics = {
            'area_plantcv': area_plantcv,
            'perimeter': float(perimeter_opencv),
            'solidity': float(solidity_opencv)
        }
        
        metrics_list.append(metrics)
    
    return metrics_list


def save_comprehensive_visualization(original_bgr: np.ndarray, 
                                   mask_plants: np.ndarray,
                                   mask_leaves: np.ndarray,
                                   roi_coords: Tuple[int, int, int, int],
                                   analysis_results: Dict[str, Any],
                                   paleta_coords: Tuple[int, int, int, int],
                                   save_path: str) -> None:
    """Visualización del proceso de segmentación - solo imágenes del proceso."""
    
    # Crear máscara de color para visualización
    leaves_colored = np.zeros_like(original_bgr)
    leaves_colored[mask_leaves > 0] = (0, 255, 0)      # Verde para hojas
    
    # Tamaño de figura más grande para máxima claridad
    fig = plt.figure(figsize=(32, 24), dpi=200)
    
    # 1. Imagen original con recuadro de paleta
    ax1 = fig.add_subplot(2, 3, 1)
    img_with_paleta = original_bgr.copy()
    # Dibujar recuadro de la paleta en magenta
    cv2.rectangle(img_with_paleta, (paleta_coords[0], paleta_coords[1]), 
                  (paleta_coords[2], paleta_coords[3]), (255, 0, 255), 3)
    # Agregar etiqueta "PALETA"
    cv2.putText(img_with_paleta, "PALETA", (paleta_coords[0], paleta_coords[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    # Mostrar coordenadas
    coords_text = f"({paleta_coords[0]},{paleta_coords[1]}) - ({paleta_coords[2]},{paleta_coords[3]})"
    cv2.putText(img_with_paleta, coords_text, (paleta_coords[0], paleta_coords[3] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    ax1.imshow(cv2.cvtColor(img_with_paleta, cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen Original + Recuadro Paleta", fontsize=20)
    ax1.axis("off")
    
    # 2. Dos ROIs circulares definidos
    ax2 = fig.add_subplot(2, 3, 2)
    roi_vis = original_bgr.copy()
    (x1_1, y1_1, x2_1, y2_1, center_x1, center_y1, radius1,
     x1_2, y1_2, x2_2, y2_2, center_x2, center_y2, radius2) = roi_coords
    
    # Dibujar círculo superior (primera maceta) en azul
    cv2.circle(roi_vis, (center_x1, center_y1), radius1, (255, 0, 0), 3)
    cv2.circle(roi_vis, (center_x1, center_y1), 5, (255, 255, 0), -1)  # Centro amarillo
    
    # Dibujar círculo inferior (segunda maceta) en verde
    cv2.circle(roi_vis, (center_x2, center_y2), radius2, (0, 255, 0), 3)
    cv2.circle(roi_vis, (center_x2, center_y2), 5, (0, 255, 255), -1)  # Centro cian
    
    # Agregar etiquetas
    cv2.putText(roi_vis, "Maceta 1", (center_x1 - 30, center_y1 - radius1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(roi_vis, "Maceta 2", (center_x2 - 30, center_y2 - radius2 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # También mostrar el recuadro de la paleta en esta vista
    cv2.rectangle(roi_vis, (paleta_coords[0], paleta_coords[1]), 
                  (paleta_coords[2], paleta_coords[3]), (255, 0, 255), 2)
    
    ax2.imshow(cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title("Dos ROIs Circulares + Paleta", fontsize=20)
    ax2.axis("off")
    
    # 3. Máscara combinada (ExG + HSV)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(mask_plants, cmap='gray')
    ax3.set_title("Máscara Combinada (ExG + HSV)", fontsize=20)
    ax3.axis("off")
    
    # 4. Hojas detectadas
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(mask_leaves, cmap='gray')
    ax4.set_title("Hojas Detectadas", fontsize=20)
    ax4.axis("off")
    
    # 5. Hojas detectadas (Verde)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(cv2.cvtColor(leaves_colored, cv2.COLOR_BGR2RGB))
    ax5.set_title("Hojas Detectadas (Verde)", fontsize=20)
    ax5.axis("off")
    
    # 6. Superposición en imagen original
    ax6 = fig.add_subplot(2, 3, 6)
    overlay = cv2.addWeighted(original_bgr, 0.6, leaves_colored, 0.4, 0)
    # Mostrar también el recuadro de la paleta en la superposición
    cv2.rectangle(overlay, (paleta_coords[0], paleta_coords[1]), 
                  (paleta_coords[2], paleta_coords[3]), (255, 0, 255), 2)
    cv2.putText(overlay, "PALETA", (paleta_coords[0], paleta_coords[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    ax6.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax6.set_title("Superposición + Paleta", fontsize=20)
    ax6.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)

def process_image_arriba(image_path: str) -> Dict[str, Any]:
    """Procesa una imagen de vista superior para detectar hojas verdes."""
    print(f"  Procesando: {os.path.basename(image_path)}")
    
    # Cargar imagen
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"     Error: No se pudo cargar la imagen")
        return {}
    
    print("     Aplicando preprocesamiento avanzado...")
    
    # 1. PREPROCESAMIENTO
    bgr_preprocessed, roi_carta, paleta_coords = apply_advanced_preprocessing(bgr)
    
    # 2. DEFINICIÓN DE ROI
    roi_mask, roi_coords = define_plant_roi_arriba(bgr_preprocessed)
    
    # 3. CALIBRACIÓN DE DETECCIÓN DE VERDE
    hsv_ranges = calibrate_green_detection(bgr_preprocessed, roi_carta)
    
    print("     Segmentando hojas verdes...")
    
    # 4. SEGMENTACIÓN MEJORADA
    mask_plants, analysis_results = enhanced_green_segmentation(
        bgr_preprocessed, roi_mask, hsv_ranges
    )
    
    # 5. USAR SOLO HOJAS
    mask_leaves = mask_plants.copy()
    
    # 6. CÁLCULO DE MÉTRICAS MORFOLÓGICAS
    morphological_metrics = calculate_morphological_metrics(mask_leaves)
    
    print("     Guardando visualización comprehensiva...")
    
    # 7. GUARDAR VISUALIZACIÓN
    stem = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(OUTPUT_DIR, PROCESSED_DIR, f"{stem}_hojas_verdes.jpg")
    
    save_comprehensive_visualization(
        bgr_preprocessed, mask_plants, mask_leaves,
        roi_coords, analysis_results, paleta_coords, vis_path
    )
    
    print(f"     Visualización guardada: {vis_path}")
    
    # 8. PREPARAR DATOS PARA CSV MORFOLÓGICO
    timestamp = extract_timestamp(stem)
    
    # Crear fila para cada hoja individual
    rows_data = []
    for i, metrics in enumerate(morphological_metrics):
        row_data = {
            "imagen": stem,
            "timestamp": timestamp,
            "hoja_id": i + 1,
            "numero_hojas_total": len(morphological_metrics),
            "area_plantcv": metrics['area_plantcv'],
            "perimetro_opencv": metrics['perimeter'],
            "solidez_opencv": metrics['solidity']
         
           
        }
        rows_data.append(row_data)
    
    # Si no hay hojas, crear una fila con valores 0
    if not morphological_metrics:
        row_data = {
            "imagen": stem,
            "timestamp": timestamp,
            "hoja_id": 0,
            "numero_hojas_total": 0,
            "area_plantcv": 0.0,
            "perimetro_opencv": 0.0,
            "solidez_opencv": 0.0
        }
        rows_data.append(row_data)
    
    return rows_data

def main_arriba():
    """Función principal para procesar imágenes de vista superior."""
    ensure_dirs()
    
    # Listar imágenes
    images = list_images(INPUT_DIR)
    if not images:
        raise FileNotFoundError(f"No hay imágenes en {INPUT_DIR}")
    
    print(f" PROCESAMIENTO DE IMÁGENES DE VISTA SUPERIOR")
    print(f" Entrada: {INPUT_DIR}")
    print(f" Salida: {OUTPUT_DIR}/{PROCESSED_DIR}")
    print(f" CSV: {CSV_PATH}")
    print("=" * 60)
    print(" PARÁMETROS DE SEGMENTACIÓN:")
    print(f"   - Área mínima hoja: {MIN_LEAF_AREA} px")
    print(f"   - H verde: {H_MIN_GREEN}-{H_MAX_GREEN}")
    print(f"   - S mínima: {S_MIN_GREEN}")
    print(f"   - V mínima: {V_MIN_GREEN}")
    print("=" * 60)
    print(" DOS ROIs CIRCULARES OPTIMIZADOS: Maceta 1 (22% alto, 59% ancho, radio 17%) y Maceta 2 (45% alto, 59% ancho, radio 16%)")
    print(" PALETA DE COLORES: Detectada en la parte inferior central (20%-80% ancho, 80%-98% alto)")
    print(" COORDENADAS PALETA: (x1, y1) - (x2, y2) donde x1=20% ancho, y1=80% alto, x2=80% ancho, y2=98% alto")
    print(" MÉTRICAS EXTRACTADAS: Área PlantCV, Perímetro OpenCV, Solidez OpenCV")
    print(" DETECCIÓN: Solo hojas - SEGMENTACIÓN EQUILIBRADA")
    print("=" * 60)
    
    # Procesar imágenes
    all_rows = []
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {os.path.basename(image_path)}")
        try:
            resultados = process_image_arriba(image_path)
            if resultados:
                all_rows.extend(resultados)
                num_hojas = resultados[0]['numero_hojas_total'] if resultados else 0
                print(f"   Hojas detectadas: {num_hojas}")
            else:
                print("   Sin resultados")
        except Exception as e:
            print(f"   Error: {e}")
            # Crear fila de error
            error_row = {
                "imagen": os.path.basename(image_path),
                "timestamp": None,
                "hoja_id": 0,
                "numero_hojas_total": 0,
                "area_plantcv": 0.0,
                "perimetro_opencv": 0.0,
                "solidez_opencv": 0.0
            }
            all_rows.append(error_row)
    
    # Guardar CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        
        print(f"\n PROCESAMIENTO COMPLETADO!")
        print(f"   - CSV: {CSV_PATH}")
        print(f"   - Visualizaciones: {os.path.join(OUTPUT_DIR, PROCESSED_DIR)}")
        
        # Resumen
        print(f"\n RESUMEN:")
        print(f"   - Total de imágenes: {len(images)}")
        print(f"   - Total de hojas analizadas: {len(all_rows)}")
        print(f"   - Hojas con métricas morfológicas: {len([r for r in all_rows if r['area_plantcv'] > 0])}")
        
        if len(all_rows) > 0:
            areas_plantcv = [r['area_plantcv'] for r in all_rows if r['area_plantcv'] > 0]
            if areas_plantcv:
                print(f"   - Área promedio (PlantCV): {np.mean(areas_plantcv):.1f} px")
                    
    else:
        print("\n No se generaron resultados")

if __name__ == "__main__":
    print(" PROCESAMIENTO DE VISTA SUPERIOR - SEGMENTACIÓN DE HOJAS VERDES")
    print("=" * 80)
    print("Este script incluye:")
    print(" Preprocesamiento avanzado (balance de blancos, CLAHE, filtro bilateral)")
    print(" Segmentación multi-método (ExG + HSV calibrado)")
    print(" Detección especializada de hojas verdes")
    print(" Visualización mejorada de 6 paneles (solo imágenes del proceso)")
    print(" MÉTRICAS MORFOLÓGICAS:")
    print("   - Área PlantCV, Perímetro OpenCV, Solidez OpenCV")
    print("   - Número de hojas, ID de hoja, Timestamp")
    print("=" * 80)
    print(" PROCESANDO: img_web_arriba (imágenes de vista superior)")
    print(" SALIDA: output/imagenes_procesadas")
    print(" CSV: output/metricas_morfologicas.csv")
    print(" MÉTODO: Índice ExG + Filtros HSV + Clasificación por forma + OpenCV + PlantCV")
    print(" ROI: Dos círculos separados para Maceta 1 (25% alto) y Maceta 2 (55% alto)")
    print("=" * 80)
    
    try:
        main_arriba()
    except Exception as e:
        print(f"\n Error en el procesamiento principal: {e}")
        print("\n Para usar el script:")
        print("   1. Coloca tus imágenes en la carpeta 'img_web_arriba'")
        print("   2. Ejecuta: python procesamiento_arriba.py")
        print("   3. Los resultados se guardarán en 'output/imagenes_procesadas'")
        print("   4. Las métricas morfológicas se guardarán en 'output/metricas_morfologicas.csv'")

