import os
import csv
import cv2
import numpy as np
from datetime import datetime
from segment_anything import sam_model_registry, SamPredictor
import plantcv as pcv

# --- Configuration ---
IMAGE_FOLDER = "img_web_pi_lateral"
OUTPUT_FOLDER = "output/imagenes_procesadas_laterales"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
CSV_PATH = "output/metricas_tallos.csv"

# Configuración para procesamiento imagen por imagen
CURRENT_IMAGE = "foto_2025-09-09_16-00-42.jpg" 

# Global variables for interactive mode
input_points = []
input_labels = []
manual_mode = True

# --- Helper Functions ---
def preprocess_image(image_bgr):
    """
    Preprocesar imagen con suavizado selectivo, corrección de dominancia de color y CLAHE
    """
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

def create_roi_from_mask(mask, padding=15):
    """
    Crear ROI (Region of Interest) basada en la máscara de la planta con filtrado inteligente
    """
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

def apply_roi_to_image(image, roi):
    """
    Aplicar ROI a una imagen
    """
    if roi is None:
        return image
    
    x_start, y_start, x_end, y_end = roi
    return image[y_start:y_end, x_start:x_end]

def clean_mask(mask):
    """
    Limpiar la máscara usando operaciones morfológicas y suavizado preservando bordes
    """
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

def show_points_on_image(image_to_show):
    temp_image = image_to_show.copy()
    for i, point in enumerate(input_points):
        x, y = int(point[0]), int(point[1])
        label = input_labels[i]
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(temp_image, (x, y), 5, color, -1)
    cv2.imshow("Click para segmentar (s=segmentar, r=reset, q=salir)", temp_image)

def mouse_callback(event, x, y, flags, param):
    global input_points, input_labels, manual_mode
    if manual_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Punto positivo anadido en: ({x}, {y})")
            input_points.append([x, y])
            input_labels.append(1)
            show_points_on_image(param)
        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"Punto negativo anadido en: ({x}, {y})")
            input_points.append([x, y])
            input_labels.append(0)
            show_points_on_image(param)

# --- Metric Functions using PlantCV ---
def get_plantcv_metrics(image_rgb, mask, image_bgr):
    """
    Función para calcular altura total y altura de cada tallo individual
    """
    # Convertir la máscara a formato compatible con PlantCV
    mask_uint8 = (mask * 255).astype(np.uint8)

    try:
        # Calcular altura total usando PlantCV
        height_pixels, width_pixels = pcv.bounding_rectangle(mask_uint8)
        
        # Análisis de tallos individuales usando detección de color
        stem_heights = analyze_individual_stems(mask_uint8, image_bgr)
        
        return {
            'total_height_pixels': height_pixels,
            'stem_heights': stem_heights
        }
        
    except Exception as e:
        print(f"Error en análisis PlantCV: {e}")
        # Fallback a OpenCV si PlantCV falla
        return get_opencv_metrics_fallback(mask)

def analyze_individual_stems(mask, image_bgr):
    """
    Analizar cada tallo individual usando ROI y detección de color verde
    """
    try:
        # Limpiar la máscara primero
        mask_cleaned = clean_mask(mask)
        
        # Crear ROI basada en la máscara
        roi = create_roi_from_mask(mask_cleaned, padding=30)
        
        if roi is None:
            print("No se pudo crear ROI, usando análisis completo")
            return analyze_without_roi(mask_cleaned)
        
        # Aplicar ROI a la imagen y máscara
        image_roi = apply_roi_to_image(image_bgr, roi)
        mask_roi = apply_roi_to_image(mask_cleaned, roi)
        
        # Preprocesar la imagen ROI
        image_enhanced_roi = preprocess_image(image_roi)
        
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
        combined_mask_roi = clean_mask(combined_mask_roi)
        
        # Encontrar contornos de tallos en ROI
        contours, _ = cv2.findContours(combined_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stem_heights = []
        min_stem_area = 80  # Área mínima muy estricta para ROI
        
        # Filtrar contornos por área y relación de aspecto
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_stem_area:
                x, y, w, h = cv2.boundingRect(c)
                # Filtro adicional: relación de aspecto (altura/ancho) debe ser > 1.5
                # Esto elimina componentes muy anchos o muy pequeños
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio > 1.5 and h > 30:  # Altura mínima de 30 píxeles
                    stem_heights.append(h)
        
        # Si no se encontraron tallos en ROI, usar método alternativo
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
        
        # Si aún no hay tallos, usar altura total de ROI
        if not stem_heights:
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if h > 30:  # Solo usar si la altura es significativa
                    stem_heights = [h]
        
        # Filtro final: eliminar tallos muy pequeños
        stem_heights = [h for h in stem_heights if h >= 40]
        
        return stem_heights
        
    except Exception as e:
        print(f"Error en análisis de tallos con ROI: {e}")
        # Fallback sin ROI
        return analyze_without_roi(mask)

def analyze_without_roi(mask_cleaned):
    """
    Análisis de tallos sin ROI (método de respaldo)
    """
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

def get_opencv_metrics_fallback(mask):
    """
    Función de respaldo usando solo OpenCV - altura total y alturas individuales
    """
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

def save_metrics_to_csv(filename, image_file, metrics_dict):
    """
    Guardar métricas con una fila por cada tallo individual, evitando duplicados
    """
    headers = [
        'Nombre_Archivo', 
        'Tallo_ID',
        'Altura_Tallo_Pixeles',
        'Timestamp'
    ]
    
    # Leer datos existentes si el archivo existe
    existing_data = []
    if os.path.exists(filename):
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_data = list(reader)
    
    # Filtrar datos existentes, excluyendo la imagen actual
    filtered_data = []
    if existing_data:
        # Actualizar headers si es necesario
        if len(existing_data[0]) < 4:  # Si no tiene timestamp
            filtered_data = [headers]  # Usar headers nuevos
        else:
            filtered_data = [existing_data[0]]  # Mantener headers existentes
        
        for row in existing_data[1:]:
            if len(row) > 0 and row[0] != image_file:
                # Asegurar que las filas antiguas tengan el formato correcto
                if len(row) < 4:  # Si no tiene timestamp, agregar uno vacío
                    row.append('')
                filtered_data.append(row)
    
    # Si no hay datos existentes, crear con headers
    if not filtered_data:
        filtered_data = [headers]
    
    # Añadir nuevos datos para la imagen actual
    stem_heights = metrics_dict['stem_heights']
    
    # Extraer fecha del nombre del archivo (formato: foto_YYYY-MM-DD_HH-MM-SS.jpg)
    try:
        # Buscar patrón de fecha en el nombre del archivo
        import re
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', image_file)
        if date_match:
            timestamp = date_match.group(1)  # Extraer YYYY-MM-DD
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d")  # Fallback a fecha actual
    except:
        timestamp = datetime.now().strftime("%Y-%m-%d")  # Fallback a fecha actual
    
    for i, height in enumerate(stem_heights, 1):
        filtered_data.append([
            image_file,
            i,  # Tallo_ID (1, 2, 3, etc.)
            height,
            timestamp
        ])
    
    # Escribir todos los datos actualizados
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_data)
    
    print(f"Métricas guardadas para {image_file}")
    print(f"  - Altura total: {metrics_dict['total_height_pixels']} px")
    print(f"  - Número de tallos: {len(stem_heights)}")
    for i, height in enumerate(stem_heights, 1):
        print(f"  - Tallo {i}: {height} px")

# --- Main Script ---
print("Cargando el modelo SAM...")
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)
print("Modelo SAM cargado.")

# Crear carpeta de salida si no existe
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Carpeta de salida creada: {OUTPUT_FOLDER}")

# Verificar que la imagen especificada existe
image_path = os.path.join(IMAGE_FOLDER, CURRENT_IMAGE)
if not os.path.exists(image_path):
    print(f"Error: La imagen {CURRENT_IMAGE} no existe en la carpeta {IMAGE_FOLDER}")
    print("Imágenes disponibles:")
    available_images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img in available_images:
        print(f"  - {img}")
    exit()

print(f"Procesando imagen: {CURRENT_IMAGE}")
print("=" * 50)

# Cargar la imagen
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

# Preprocesar la imagen
print("Preprocesando imagen (CLAHE + Corrección dominancia color + Suavizado selectivo + ROI)...")
image_enhanced = preprocess_image(image_bgr)
image_rgb = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB)

# Configurar ventana para segmentación manual
cv2.namedWindow("Click para segmentar (s=segmentar, r=reset, q=salir)")
cv2.setMouseCallback("Click para segmentar (s=segmentar, r=reset, q=salir)", mouse_callback, image_bgr)

print("Instrucciones:")
print("- Clic izquierdo: Punto positivo (parte de la planta)")
print("- Clic derecho: Punto negativo (fondo)")
print("- 's': Segmentar")
print("- 'r': Resetear puntos")
print("- 'q': Salir")

# Bucle de segmentación manual
while manual_mode:
    cv2.imshow("Click para segmentar (s=segmentar, r=reset, q=salir)", image_bgr)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if not input_points:
            print("No se han añadido puntos. Haz clic para empezar.")
            continue
            
        print("Segmentando...")
        predictor.set_image(image_rgb)
        input_points_np = np.array(input_points)
        input_labels_np = np.array(input_labels)
        
        masks, scores, _ = predictor.predict(
            point_coords=input_points_np,
            point_labels=input_labels_np,
            multimask_output=False,
        )
        
        final_mask = masks[0]
        manual_mode = False
        cv2.destroyAllWindows()
        break
        
    elif key == ord('q'):
        print("Saliendo...")
        cv2.destroyAllWindows()
        exit()
        
    elif key == ord('r'):
        print("Puntos reseteados.")
        input_points.clear()
        input_labels.clear()

# Procesar y guardar resultados
print("-" * 50)
print("Analizando métricas con PlantCV...")

# Extraer métricas usando PlantCV
metrics = get_plantcv_metrics(image_rgb, final_mask, image_enhanced)

# Guardar métricas en CSV
save_metrics_to_csv(CSV_PATH, CURRENT_IMAGE, metrics)

# Crear imagen de salida con la máscara
mask_color = np.zeros_like(image_bgr, dtype=np.uint8)
mask_color[final_mask > 0.0] = [0, 255, 255]  # Amarillo para la máscara
output_image = cv2.addWeighted(image_bgr, 0.7, mask_color, 0.3, 0)
        
# Añadir puntos de segmentación a la imagen de salida
for i, point in enumerate(input_points):
    x, y = int(point[0]), int(point[1])
    label = input_labels[i]
    color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Verde para positivos, rojo para negativos
    cv2.circle(output_image, (x, y), 5, color, -1)

# Guardar imagen segmentada
output_path = os.path.join(OUTPUT_FOLDER, CURRENT_IMAGE)
cv2.imwrite(output_path, output_image)
print(f"Imagen segmentada guardada en: {output_path}")

print("=" * 50)
print("Proceso completado exitosamente!")
print(f"Para procesar otra imagen, cambia la variable CURRENT_IMAGE en el código")
print(f"y ejecuta el script nuevamente.")