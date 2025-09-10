import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from typing import Dict, List, Tuple, Any
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Constantes del sistema
INPUT_DIR = "img_noir"
OUTPUT_DIR = "output"
PROCESSED_DIR = "imagenes_procesadas_noir_avanzado"
CSV_PATH = os.path.join(OUTPUT_DIR, "metricas_reflectancia.csv")

# Parámetros de segmentación para hojas verdes en NIR
MIN_LEAF_AREA = 350        # Área mínima para eliminar objetos muy pequeños
MIN_STEM_AREA = 50           # Área mínima para tallos
KERNEL_SIZE = 3              # Tamaño del kernel para morfología



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
    # Patrón más flexible para nombres como "foto_2025-08-27_16-00-10"
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

def apply_advanced_preprocessing_nir(bgr: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento avanzado para imágenes NIR con enfoque en plantas verdes:
    - Balance de blancos Gray World
    - CLAHE para contraste en canal V
    - Ajuste de luminosidad para mejor detección de hojas
    - Suavizado selectivo para preservar detalles de hojas
    """
    try:
        h, w = bgr.shape[:2]
        
        # 1. BALANCE DE BLANCOS GRAY WORLD (simplificado)
        # Convertir a float32 para cálculos
        bgr_float = bgr.astype(np.float32)
        b, g, r = bgr_float[:,:,0], bgr_float[:,:,1], bgr_float[:,:,2]
        
        # Calcular medias
        mb, mg, mr = np.mean(b) + 1e-6, np.mean(g) + 1e-6, np.mean(r) + 1e-6
        k = (mb + mg + mr) / 3.0
        
        # Aplicar balance
        b_balanced = np.clip(b * k/mb, 0, 255)
        g_balanced = np.clip(g * k/mg, 0, 255)
        r_balanced = np.clip(r * k/mr, 0, 255)
        
        # Reconstruir imagen
        bgr_balanced = np.stack([b_balanced, g_balanced, r_balanced], axis=2).astype(np.uint8)
        
        # 2. CLAHE PARA CONTRASTE EN CANAL V
        hsv = cv2.cvtColor(bgr_balanced, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # CLAHE solo en el canal V (brillo) para preservar colores
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        
        # 3. AJUSTE DE LUMINOSIDAD PARA MEJOR DETECCIÓN DE HOJAS
        # Aumentar levemente la luminosidad del canal V para destacar hojas
        v_brightened = np.clip(v_enhanced * 1.15, 0, 255).astype(np.uint8)  # +15% de luminosidad
        
        # Reconstruir HSV con canal V mejorado y más brillante
        hsv_enhanced = np.stack([h, s, v_brightened], axis=2)
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 4. SUAVIZADO SELECTIVO (bilateral filter para preservar bordes)
        bgr_smooth = cv2.bilateralFilter(bgr_enhanced, 9, 75, 75)
        
        return bgr_smooth
        
    except Exception as e:
        print(f"  ⚠️ Error en preprocesamiento avanzado NIR: {e}")
        # Fallback: imagen original
        return bgr

def define_plant_roi_noir(bgr: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Define ROIs circulares adaptativos para plantas en imágenes NIR.
    Combina detección automática con coordenadas base como respaldo.
    """
    h, w = bgr.shape[:2]
    
    print("    🔧 Usando ROIs adaptativos con detección automática...")
    
    # Coordenadas base como respaldo (las originales)
    TAMANO_ESTANDAR_ROI = 63  # Tamaño estándar basado en esta imagen específica
    
    # Coordenadas base para las dos macetas
    coordenadas_base = [
        (625, 295, TAMANO_ESTANDAR_ROI),  # Maceta superior (amarilla) - calibrada para maceta negra real
        (641, 455, TAMANO_ESTANDAR_ROI)   # Maceta inferior (rosada) - calibrada para maceta negra real
    ]
    
    print(f"       Coordenadas base:")
    print(f"         Maceta 1 (ARRIBA): centro(625, 295), radio={TAMANO_ESTANDAR_ROI}px")
    print(f"         Maceta 2 (ABAJO): centro(630, 455), radio={TAMANO_ESTANDAR_ROI}px")
    
    # DETECCIÓN AUTOMÁTICA DE MACETAS
    macetas_detectadas = detect_automatic_pots(bgr, coordenadas_base)
    
    # CREACIÓN DE ROIs
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_info = []
    
    for i, (cx, cy, radius) in enumerate(macetas_detectadas):
        # Asegurar que el círculo esté dentro de los límites de la imagen
        center_x = max(radius, min(w - radius, cx))
        center_y = max(radius, min(h - radius, cy))
        
        # Dibujar círculo en la máscara
        cv2.circle(roi_mask, (center_x, center_y), radius, 255, -1)
        
        # Guardar información
        roi_info.append((center_x, center_y, radius))
        
        posicion = "ARRIBA" if i == 0 else "ABAJO"
        print(f"       Maceta {i+1} ({posicion}): centro({center_x}, {center_y}), radio={radius}px")
    
    print(f"    📏 ROIs creados: {len(roi_info)} círculos adaptativos")
    
    return roi_mask, roi_info

def detect_automatic_pots(bgr: np.ndarray, base_coordinates: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Detecta automáticamente las macetas negras cerca de las coordenadas base conocidas.
    Usa las coordenadas originales como respaldo si la detección falla.
    
    Args:
        bgr: Imagen en formato BGR
        base_coordinates: Lista de coordenadas base (cx, cy, radius)
    
    Returns:
        Lista de coordenadas ajustadas (cx, cy, radius)
    """
    h, w = bgr.shape[:2]
    
    print("     Detectando macetas automáticamente...")
    
    # Convertir a escala de grises para detección
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral para detectar objetos oscuros (macetas negras)
    # Las macetas negras tienen valores bajos en escala de grises
    _, mask_dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos de objetos oscuros
    contours, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    adjusted_coordinates = []
    
    for i, (base_cx, base_cy, base_radius) in enumerate(base_coordinates):
        posicion = "ARRIBA" if i == 0 else "ABAJO"
        print(f"       🔍 Analizando Maceta {i+1} ({posicion})...")
        
        # Buscar contornos cerca de las coordenadas base
        best_contour = None
        best_score = 0
        search_radius = base_radius * 2  # Radio de búsqueda alrededor de las coordenadas base
        
        for contour in contours:
            # Calcular el centro del contorno
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Verificar si está dentro del radio de búsqueda
            distance = np.sqrt((cx - base_cx)**2 + (cy - base_cy)**2)
            if distance > search_radius:
                continue
            
            # Calcular área del contorno
            area = cv2.contourArea(contour)
            
            # Verificar que el área sea razonable para una maceta
            expected_area = np.pi * base_radius**2
            area_ratio = area / expected_area
            
            # Calcular circularidad
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Calcular puntuación basada en:
            # 1. Distancia a coordenadas base (más cerca = mejor)
            # 2. Circularidad (más circular = mejor)
            # 3. Relación de área (más cercana a 1 = mejor)
            distance_score = 1.0 / (1.0 + distance / base_radius)
            circularity_score = circularity
            area_score = 1.0 / (1.0 + abs(area_ratio - 1.0))
            
            total_score = distance_score * 0.4 + circularity_score * 0.4 + area_score * 0.2
            
            if total_score > best_score and circularity > 0.3 and 0.3 < area_ratio < 3.0:
                best_score = total_score
                best_contour = contour
        
        if best_contour is not None and best_score > 0.5:
            # Calcular el centro y radio del mejor contorno
            M = cv2.moments(best_contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calcular radio basado en el área
            area = cv2.contourArea(best_contour)
            radius = int(np.sqrt(area / np.pi))
            
            # Asegurar que el radio esté dentro de límites razonables
            radius = max(50, min(100, radius))
            
            adjusted_coordinates.append((cx, cy, radius))
            print(f"        Maceta {i+1} ({posicion}): detectada automáticamente - centro({cx}, {cy}), radio={radius}px (score: {best_score:.2f})")
        else:
            # Usar coordenadas base como respaldo
            adjusted_coordinates.append((base_cx, base_cy, base_radius))
            print(f"        Maceta {i+1} ({posicion}): usando coordenadas base - centro({base_cx}, {base_cy}), radio={base_radius}px")
    
    print(f"    🔍 Detección automática completada: {len(adjusted_coordinates)} macetas analizadas")
    return adjusted_coordinates


def enhanced_green_segmentation_nir(gray_preprocessed: np.ndarray, roi_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Segmentación multi-nivel usando múltiples umbrales de reflectancia NIR:
    - Prueba diferentes percentiles ( 90, 91.5)
    - Combina las mejores detecciones de cada nivel
    - Valida la calidad y cobertura de hojas
    - Mantiene la precisión sin introducir ruido
    """
    try:
        h, w = gray_preprocessed.shape[:2]
        
        # 0. ANÁLISIS PRELIMINAR DE LA IMAGEN PREPROCESADA
        mean_intensity = np.mean(gray_preprocessed)
        std_intensity = np.std(gray_preprocessed)
        
        print(f"     Análisis de imagen preprocesada: intensidad={mean_intensity:.1f}, desv_std={std_intensity:.1f}")
        
        # 1. ANÁLISIS DE REFLECTANCIA NIR DENTRO DEL ROI
        roi_coords = np.where(roi_mask > 0)
        if len(roi_coords[0]) == 0:
            print(f"     ROI vacío, no se puede procesar")
            return np.zeros_like(gray_preprocessed), {}
        
        roi_intensities = gray_preprocessed[roi_coords[0], roi_coords[1]]
        
        if len(roi_intensities) == 0:
            print(f"     No hay intensidades en ROI")
            return np.zeros_like(gray_preprocessed), {}
        
        # 2. SEGMENTACIÓN MULTI-NIVEL CON MÚLTIPLES UMBRALES
        print(f"    🔧 Iniciando segmentación multi-nivel...")
        
        # Definir percentiles a probar (de menos a más estricto)
        percentiles = [  89.8,90 , 91.5]
        masks_by_percentile = {}
        scores_by_percentile = {}
        
        for percentile in percentiles:
            try:
                # Calcular umbral para este percentil
                threshold = np.percentile(roi_intensities, percentile)
                
                # Crear máscara binaria
                mask_binary = cv2.threshold(gray_preprocessed, threshold, 255, cv2.THRESH_BINARY)[1]
                mask_roi = cv2.bitwise_and(mask_binary, roi_mask)
                
                # Postprocesamiento morfológico
                kernel = np.ones((3, 3), np.uint8)
                mask_clean = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
                
                # Filtrar por tamaño y forma
                mask_filtered = clean_components_by_size_and_shape(mask_clean, MIN_LEAF_AREA)
                
                # Calcular métricas de calidad
                total_pixels = np.count_nonzero(mask_filtered)
                coverage_percentage = total_pixels / mask_filtered.size * 100
                
                # Calcular puntuación de calidad basada en:
                # 1. Cobertura (más píxeles = mejor, pero no excesivo)
                # 2. Consistencia del umbral (percentil más alto = más confiable)
                # 3. Relación con intensidad media del ROI
                roi_mean = np.mean(roi_intensities)
                threshold_ratio = threshold / roi_mean if roi_mean > 0 else 1.0
                
                # Puntuación de cobertura (óptima entre 0.1% y 2%)
                coverage_score = 1.0 / (1.0 + abs(coverage_percentage - 0.5))
                
                # Puntuación de confiabilidad (percentil más alto = mejor)
                reliability_score = percentile / 100.0
                
                # Puntuación de umbral (cerca de la media del ROI = mejor)
                threshold_score = 1.0 / (1.0 + abs(threshold_ratio - 1.2))
                
                # Puntuación total ponderada
                total_score = coverage_score * 0.4 + reliability_score * 0.3 + threshold_score * 0.3
                
                masks_by_percentile[percentile] = mask_filtered
                scores_by_percentile[percentile] = {
                    'score': total_score,
                    'threshold': threshold,
                    'pixels': total_pixels,
                    'coverage': coverage_percentage,
                    'threshold_ratio': threshold_ratio
                }
                
                print(f"       Percentil {percentile}: umbral={threshold:.1f}, píxeles={total_pixels}, cobertura={coverage_percentage:.3f}%, score={total_score:.3f}")
                
            except Exception as e:
                print(f"        Error en percentil {percentile}: {e}")
                continue
        
        if not masks_by_percentile:
            print(f"     No se pudo generar ninguna máscara válida")
            return np.zeros_like(gray_preprocessed), {}
        
        # 3. SELECCIÓN INTELIGENTE DE LA MEJOR COMBINACIÓN
        print(f"    🔧 Seleccionando mejor combinación de umbrales...")
        
        # Ordenar por puntuación
        sorted_percentiles = sorted(scores_by_percentile.keys(), 
                                  key=lambda p: scores_by_percentile[p]['score'], 
                                  reverse=True)
        
        # Tomar el mejor percentil como base
        best_percentile = sorted_percentiles[0]
        best_mask = masks_by_percentile[best_percentile]
        best_score = scores_by_percentile[best_percentile]
        
        print(f"       Mejor percentil: {best_percentile} (score: {best_score['score']:.3f})")
        
        # 4. COMBINACIÓN INTELIGENTE CON OTROS UMBRALES
        combined_mask = best_mask.copy()
        additional_pixels = 0
        
        # Probar agregar píxeles de otros percentiles si mejoran la cobertura
        for percentile in sorted_percentiles[1:]:
            if scores_by_percentile[percentile]['score'] > 0.6:  # Solo si es de buena calidad
                current_mask = masks_by_percentile[percentile]
                
                # Encontrar píxeles adicionales que no estén en la máscara combinada
                additional_pixels_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(combined_mask))
                additional_pixels_count = np.count_nonzero(additional_pixels_mask)
                
                # Solo agregar si hay píxeles adicionales significativos
                if additional_pixels_count > 50:  # Mínimo 50 píxeles adicionales
                    combined_mask = cv2.bitwise_or(combined_mask, additional_pixels_mask)
                    additional_pixels += additional_pixels_count
                    print(f"       Agregando percentil {percentile}: +{additional_pixels_count} píxeles")
        
        if additional_pixels > 0:
            print(f"       Total píxeles adicionales agregados: {additional_pixels}")
        
        # 5. LIMPIEZA FINAL DE LA MÁSCARA COMBINADA
        final_mask = clean_components_by_size_and_shape(combined_mask, MIN_LEAF_AREA)
        
        # 6. ANÁLISIS DE RESULTADOS FINALES
        final_pixels = np.count_nonzero(final_mask)
        final_coverage = final_pixels / final_mask.size * 100
        
        analysis_results = {
            'best_percentile': best_percentile,
            'best_threshold': best_score['threshold'],
            'best_score': best_score['score'],
            'final_pixels': final_pixels,
            'final_coverage': final_coverage,
            'additional_pixels_added': additional_pixels,
            'percentiles_tested': list(percentiles),
            'scores_by_percentile': scores_by_percentile,
            'min_leaf_area': MIN_LEAF_AREA,
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'roi_pixels': len(roi_intensities),
            'roi_mean_intensity': float(np.mean(roi_intensities))
        }
        
        print(f"     Segmentación multi-nivel completada:")
        print(f"       - Mejor percentil: {best_percentile} (umbral: {best_score['threshold']:.1f})")
        print(f"       - Píxeles finales: {final_pixels}")
        print(f"       - Cobertura final: {final_coverage:.3f}%")
        print(f"       - Píxeles adicionales: {additional_pixels}")
        
        return final_mask, analysis_results
        
    except Exception as e:
        print(f"   Error en segmentación multi-nivel NIR: {e}")
        empty_mask = np.zeros_like(gray_preprocessed)
        empty_analysis = {
            'best_percentile': 0,
            'best_threshold': 0,
            'best_score': 0.0,
            'final_pixels': 0,
            'final_coverage': 0.0,
            'additional_pixels_added': 0,
            'percentiles_tested': [],
            'scores_by_percentile': {},
            'min_leaf_area': MIN_LEAF_AREA,
            'total_pixels': 0,
            'percentage': 0.0,
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'roi_pixels': 0,
            'roi_mean_intensity': 0.0
        }
        return empty_mask, empty_analysis
        
    except Exception as e:
        print(f"  ⚠️ Error en segmentación NIR: {e}")
        empty_mask = np.zeros_like(gray_preprocessed)
        empty_analysis = {
            'nir_threshold': 0,
            'min_leaf_area': MIN_LEAF_AREA,
            'total_pixels': 0,
            'percentage': 0.0,
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'roi_pixels': 0,
            'roi_mean_intensity': 0.0
        }
        return empty_mask, empty_analysis
        
    except Exception as e:
        print(f"  ⚠️ Error en segmentación mejorada NIR: {e}")
        empty_mask = np.zeros_like(bgr[:,:,0])
        empty_analysis = {
            'exg_threshold': 0,
            'hsv_ranges': {'h': (0, 0), 's': (0, 0), 'v': (0, 0)},
            'min_leaf_area': MIN_LEAF_AREA,
            'total_pixels': 0,
            'percentage': 0.0,
            'exg_mean': 0.0,
            'exg_std': 0.0,
            'hsv_flexible_used': False,
            'edges_detection': False,
            'morphology_kernel_size': 3
        }
        return empty_mask, empty_analysis

def clean_components_by_size_and_shape(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Limpieza avanzada de componentes por tamaño y forma usando contornos.
    Filtra por área, circularidad y relación de aspecto para mantener solo hojas válidas.
    Optimizada para hojas grises en condiciones de luz adversas.
    MEJORADA PARA ELIMINAR OBJETOS MUY PEQUEÑOS Y PUNTOS FALSOS VERDES.
    """
    # 1. LIMPIEZA MORFOLÓGICA PREVIA PARA ELIMINAR RUIDO PEQUEÑO
    # Usar kernel más pequeño para eliminar puntos muy pequeños
    kernel_small = np.ones((2, 2), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 2. ENCONTRAR CONTORNOS EN LA MÁSCARA LIMPIA
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask_cleaned
    
    mask_clean = np.zeros_like(mask_cleaned)
    
    # 3. FILTRADO INTELIGENTE POR TAMAÑO Y FORMA
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # FILTRO PRINCIPAL: Área mínima estricta
        if area < min_area:
            continue
        
        # FILTRO SECUNDARIO: Para objetos muy pequeños, aplicar criterios más estrictos
        if area < 20:  # Objetos entre 5-20 píxeles
            # Para objetos pequeños, verificar que tengan forma de hoja real
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Calcular circularidad
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Calcular relación de aspecto
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / float(h) if h > 0 else 0
            
            # CRITERIOS MUY ESTRICTOS para objetos pequeños (eliminar puntos falsos)
            if not (0.20 <= circularity <= 0.80 and  # Circularidad más restrictiva
                    0.40 <= aspect_ratio <= 2.5):      # Relación de aspecto más restrictiva
                continue
                
            # Verificar convexidad para objetos pequeños
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6) if hull_area > 0 else 0
            
            if solidity < 0.70:  # Solidez muy alta para objetos pequeños
                continue
        
        # Para objetos más grandes (>20 píxeles), usar criterios estándar
        else:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / float(h) if h > 0 else 0
            
            # Criterios estándar para objetos grandes
            if not (0.15 <= circularity <= 0.85 and
                    0.30 <= aspect_ratio <= 3.5):
                continue
                
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6) if hull_area > 0 else 0
            
            if solidity < 0.60:
                continue
        
        # Si pasa todos los filtros, mantener el contorno
        cv2.fillPoly(mask_clean, [contour], 255)
    
    # 4. LIMPIEZA FINAL ADICIONAL
    # Aplicar una apertura final para eliminar cualquier ruido residual
    kernel_final = np.ones((3, 3), np.uint8)
    mask_final = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_final, iterations=1)
    
    return mask_final

def calculate_ndvi_metrics_nir(mask_leaves: np.ndarray, original_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Calcula métricas de reflectancia NDVI para cada hoja detectada en imágenes NIR:
    - Área de la hoja
    - NDVI promedio
    - Desviación estándar del NDVI
    """
    contours, _ = cv2.findContours(mask_leaves, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metrics_list = []
    
    # Convertir imagen a RGB para cálculo de NDVI
    rgb_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_LEAF_AREA:
            continue
        
        # Crear máscara para esta hoja específica
        leaf_mask = np.zeros_like(mask_leaves)
        cv2.fillPoly(leaf_mask, [contour], 255)
        
        # Extraer valores RGB solo de esta hoja
        rgb_values = rgb_image[leaf_mask > 0]
        
        if len(rgb_values) > 0:
            # Separar canales RGB
            r_values = rgb_values[:, 0].astype(np.float32)  # Canal Rojo
            nir_values = rgb_values[:, 1].astype(np.float32)  # Canal NIR (verde en RGB)
            
            # Calcular NDVI para cada píxel: (NIR - R) / (NIR + R)
            # Evitar división por cero
            denominator = nir_values + r_values
            ndvi_values = np.where(denominator > 0, (nir_values - r_values) / denominator, 0)
            
            # Calcular estadísticas NDVI
            ndvi_mean = float(np.mean(ndvi_values))
            ndvi_std = float(np.std(ndvi_values))
            
            metrics = {
                'area': float(area),
                'ndvi_mean': ndvi_mean,
                'ndvi_std': ndvi_std
            }
            
            metrics_list.append(metrics)
    
    return metrics_list

def save_comprehensive_visualization_nir(original_bgr: np.ndarray, 
                                        mask_plants: np.ndarray,
                                        mask_leaves: np.ndarray,
                                        roi_info: List[Tuple[int, int, int]],
                                        analysis_results: Dict[str, Any],
                                        leaf_metrics: Dict[str, Any],
                                        save_path: str) -> None:
    """Visualización de solo imágenes del proceso completo de segmentación NIR."""
    
    # Crear máscaras de color para visualización
    leaves_colored = np.zeros_like(original_bgr)
    leaves_colored[mask_leaves > 0] = (0, 255, 0)      # Verde para hojas
    
    # Tamaño de figura optimizado para 6 imágenes más grandes
    fig = plt.figure(figsize=(24, 16), dpi=150)
    
    # 1. Imagen original
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen Original NIR", fontsize=16)
    ax1.axis("off")
    
    # 2. ROIs circulares adaptativos definidos
    ax2 = fig.add_subplot(2, 3, 2)
    roi_vis = original_bgr.copy()
    
    # Coordenadas base para referencia
    coordenadas_base = [
        (625, 295, 72),  # Maceta superior
        (640, 455, 72)   # Maceta inferior
    ]
    
    # Dibujar coordenadas base como círculos punteados (referencia)
    for i, (base_cx, base_cy, base_radius) in enumerate(coordenadas_base):
        color_base = (128, 128, 128)  # Gris para coordenadas base
        cv2.circle(roi_vis, (base_cx, base_cy), base_radius, color_base, 2, cv2.LINE_8, 0)
        # Etiqueta para coordenadas base
        posicion = "ARRIBA" if i == 0 else "ABAJO"
        label_base = f"Base {i+1} ({posicion})"
        cv2.putText(roi_vis, label_base, (base_cx - 50, base_cy + base_radius + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_base, 1)
    
    # Dibujar los ROIs circulares adaptativos detectados
    for i, (center_x, center_y, radius) in enumerate(roi_info):
        color = (0, 255, 255) if i == 0 else (255, 0, 255)  # Amarillo para arriba, Magenta para abajo
        cv2.circle(roi_vis, (center_x, center_y), radius, color, 4)
        # Agregar etiqueta de maceta con posición
        posicion = "ARRIBA" if i == 0 else "ABAJO"
        label = f"Maceta {i+1} ({posicion})"
        cv2.putText(roi_vis, label, (center_x - 50, center_y - radius - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    ax2.imshow(cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title("ROIs Circulares Adaptativos (Base + Detectados)", fontsize=16)
    ax2.axis("off")
    
    # 3. Máscara de plantas detectadas (multi-nivel)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(mask_plants, cmap='gray')
    ax3.set_title("Máscara de Plantas (Multi-Nivel)", fontsize=16)
    ax3.axis("off")
    
    # 4. Hojas detectadas (máscara binaria)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(mask_leaves, cmap='gray')
    ax4.set_title("Hojas Detectadas", fontsize=16)
    ax4.axis("off")
    
    # 5. Hojas detectadas (color verde)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(cv2.cvtColor(leaves_colored, cv2.COLOR_BGR2RGB))
    ax5.set_title("Hojas Detectadas (Verde)", fontsize=16)
    ax5.axis("off")
    
    # 6. Superposición en imagen original
    ax6 = fig.add_subplot(2, 3, 6)
    overlay = cv2.addWeighted(original_bgr, 0.6, leaves_colored, 0.4, 0)
    ax6.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax6.set_title("Superposición en Original", fontsize=16)
    ax6.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)

def process_image_noir_avanzado(image_path: str) -> Dict[str, Any]:
    """Procesa una imagen NIR para detectar hojas verdes con preprocesamiento avanzado."""
    print(f"  Procesando: {os.path.basename(image_path)}")
    
    # Cargar imagen
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"    ❌ Error: No se pudo cargar la imagen")
        return {}
    
    # Convertir a uint8 si es necesario (las imágenes NIR suelen ser float64)
    if bgr.dtype != np.uint8:
        if bgr.dtype == np.float64:
            bgr = (bgr * 255).astype(np.uint8)
        elif bgr.dtype == np.float32:
            bgr = (bgr * 255).astype(np.uint8)
        else:
            bgr = bgr.astype(np.uint8)
        print(f"    🔧 Convertida imagen de {bgr.dtype} to uint8")
    
    print("    🔧 Aplicando preprocesamiento avanzado NIR...")
    
    # 1. PREPROCESAMIENTO AVANZADO
    bgr_preprocessed = apply_advanced_preprocessing_nir(bgr)
    
    # 2. CONVERTIR A ESCALA DE GRISES PARA SEGMENTACIÓN NIR
    gray_preprocessed = cv2.cvtColor(bgr_preprocessed, cv2.COLOR_BGR2GRAY)
    print("    🔧 Imagen convertida a escala de grises para análisis NIR")
    
    # 3. DEFINICIÓN DE ROI
    roi_mask, roi_info = define_plant_roi_noir(bgr_preprocessed)
    
    print("     Segmentando hojas verdes usando reflectancia NIR...")
    
    # 4. SEGMENTACIÓN SIMPLIFICADA CON REFLECTANCIA NIR
    mask_plants, analysis_results = enhanced_green_segmentation_nir(
        gray_preprocessed, roi_mask
    )
    
    # 5. USAR SOLO HOJAS (sin separar tallos)
    mask_leaves = mask_plants.copy()
    
    # 6. CÁLCULO DE MÉTRICAS NDVI
    ndvi_metrics = calculate_ndvi_metrics_nir(mask_leaves, bgr_preprocessed)
    
    print("    💾 Guardando visualización comprehensiva...")
    
    # 7. GUARDAR VISUALIZACIÓN
    stem = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(OUTPUT_DIR, PROCESSED_DIR, f"{stem}_noir_avanzado.jpg")
    
    # Usar métricas NDVI para la visualización
    leaf_metrics = {
        'numero_hojas': len(ndvi_metrics),
        'area_total_hojas': sum(m['area'] for m in ndvi_metrics),
        'area_promedio_hojas': np.mean([m['area'] for m in ndvi_metrics]) if ndvi_metrics else 0.0,
        'ndvi_promedio': np.mean([m['ndvi_mean'] for m in ndvi_metrics]) if ndvi_metrics else 0.0,
        'ndvi_desv_std': np.mean([m['ndvi_std'] for m in ndvi_metrics]) if ndvi_metrics else 0.0
    }
    
    save_comprehensive_visualization_nir(
        bgr_preprocessed, mask_plants, mask_leaves,
        roi_info, analysis_results, leaf_metrics, vis_path
    )
    
    print(f"     Visualización guardada: {vis_path}")
    
    # 8. PREPARAR DATOS PARA CSV
    timestamp = extract_timestamp(stem)
    
    # Crear fila para cada hoja individual
    rows_data = []
    for i, metrics in enumerate(ndvi_metrics):
        row_data = {
            "imagen": stem,
            "timestamp": timestamp,
            "hoja_id": i + 1,
            "numero_hojas_total": len(ndvi_metrics),
            "area": metrics['area'],
            "ndvi_mean": metrics['ndvi_mean'],
            "ndvi_std": metrics['ndvi_std']
        }
        rows_data.append(row_data)
    
    # Si no hay hojas, crear una fila con valores 0
    if not ndvi_metrics:
        row_data = {
            "imagen": stem,
            "timestamp": timestamp,
            "hoja_id": 0,
            "numero_hojas_total": 0,
            "area": 0.0,
            "ndvi_mean": 0.0,
            "ndvi_std": 0.0
        }
        rows_data.append(row_data)
    
    return rows_data

def main_noir_avanzado():
    """Función principal para procesar imágenes NIR con preprocesamiento avanzado."""
    ensure_dirs()
    
    # Listar imágenes
    images = list_images(INPUT_DIR)
    if not images:
        raise FileNotFoundError(f"No hay imágenes en {INPUT_DIR}")
    
    print(f" PROCESAMIENTO AVANZADO DE IMÁGENES NIR")
    print(f" Entrada: {INPUT_DIR}")
    print(f" Salida: {OUTPUT_DIR}/{PROCESSED_DIR}")
    print(f" CSV: {CSV_PATH}")
    print("=" * 60)
    print(" PARÁMETROS DE SEGMENTACIÓN NIR:")
    print(f"   - Área mínima hoja: {MIN_LEAF_AREA} px (ESTRICTO para eliminar puntos falsos)")
    print(f"   - Área mínima tallo: {MIN_STEM_AREA} px")
    print(f"   - Método: NDVI (sin parámetros HSV)")
    print("=" * 60)
    print(" ROIs ADAPTATIVOS: Maceta 1 (arriba, amarilla) y Maceta 2 (abajo, rosada) - Detección automática + coordenadas base como respaldo")
    print(" MÉTRICAS EXTRACTADAS: Área, NDVI promedio, NDVI desv_std")
    print(" DETECCIÓN: Solo hojas verdes (sin tallos)")
    print(" PREPROCESAMIENTO: Balance de blancos + Corrección de color + CLAHE")
    print(" SEGMENTACIÓN: Multi-nivel con múltiples umbrales + combinación inteligente")
    print(" FILTRADO MEJORADO: Eliminación estricta de objetos ")
    print("=" * 60)
    
    # Procesar imágenes
    all_rows = []
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {os.path.basename(image_path)}")
        try:
            resultados = process_image_noir_avanzado(image_path)
            if resultados:
                all_rows.extend(resultados)
                num_hojas = resultados[0]['numero_hojas_total'] if resultados else 0
                print(f"  ✅ Hojas detectadas: {num_hojas}")
            else:
                print("  ⚠️ Sin resultados")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            # Crear fila de error
            error_row = {
                "imagen": os.path.basename(image_path),
                "timestamp": None,
                "hoja_id": 0,
                "numero_hojas_total": 0,
                "area": 0.0,
                "ndvi_mean": 0.0,
                "ndvi_std": 0.0
            }
            all_rows.append(error_row)
    
    # Guardar CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        
        print(f"\n PROCESAMIENTO NIR COMPLETADO!")
        print(f"   - CSV: {CSV_PATH}")
        print(f"   - Visualizaciones: {os.path.join(OUTPUT_DIR, PROCESSED_DIR)}")
        
        # Resumen
        print(f"\n RESUMEN NIR:")
        print(f"   - Total de imágenes: {len(images)}")
        print(f"   - Total de hojas analizadas: {len(all_rows)}")
        print(f"   - Hojas con métricas morfológicas: {len([r for r in all_rows if r['area'] > 0])}")
        
        if len(all_rows) > 0:
            areas = [r['area'] for r in all_rows if r['area'] > 0]
            if areas:
                print(f"   - Área promedio de hojas: {np.mean(areas):.1f} px")
                ndvi_avg = np.mean([r['ndvi_mean'] for r in all_rows if r['ndvi_mean'] != 0])
                ndvi_std_avg = np.mean([r['ndvi_std'] for r in all_rows if r['ndvi_std'] != 0])
                print(f"   - NDVI promedio: {ndvi_avg:.3f}")
                print(f"   - NDVI desv_std promedio: {ndvi_std_avg:.3f}")
    else:
        print("\n No se generaron resultados")

if __name__ == "__main__":
    print(" PLANT CV PROCESAMIENTO NIR AVANZADO - SEGMENTACIÓN DE HOJAS VERDES")
    print("=" * 80)
    print("Este script incluye:")
    print(" Preprocesamiento avanzado (balance de blancos, corrección de color, CLAHE)")
    print(" Detección automática de macetas con coordenadas base como respaldo")
    print(" Segmentación especializada de hojas verdes en NIR")
    print(" Visualización comprehensiva de 6 paneles")
    print(" MÉTRICAS NDVI DETALLADAS (Índice de Vegetación) para análisis de reflectancia")
    print("=" * 80)
    print(" PROCESANDO: img_noir (imágenes NIR)")
    print(" SALIDA: output/imagenes_procesadas_noir_avanzado")
    print(" CSV: output/metricas_reflectancia.csv")
    print(" MÉTODO: Reflectancia NIR directa + Análisis NDVI + ROIs adaptativos")
    print(" ROIs: Maceta 1 (arriba, amarilla) y Maceta 2 (abajo, rosada) - Detección automática + coordenadas base como respaldo")
    print(" PREPROCESAMIENTO: Balance de blancos + Corrección de color + CLAHE")
    print(" SEGMENTACIÓN: Análisis directo de reflectancia NIR dentro del ROI")
    print("=" * 80)
    
    try:
        main_noir_avanzado()
    except Exception as e:
        print(f"\n Error en el procesamiento principal: {e}")
        print("\n Para usar el script:")
        print("   1. Coloca tus imágenes NIR en la carpeta 'img_noir'")
        print("   2. Ejecuta: python rrspaldo.py")
        print("   3. Los resultados se guardarán en 'output/imagenes_procesadas_noir_avanzado'")
        print("   4. Las métricas se guardarán en 'output/metricas_hsv_noir_avanzado.csv'")

       