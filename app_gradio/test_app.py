
import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import PlantPhenotypingApp, create_interface
    print("✅ Módulos importados correctamente")
    
    # Crear instancia de la aplicación
    app = PlantPhenotypingApp()
    print("✅ Aplicación creada correctamente")
    
    # Verificar que el directorio output existe
    output_dir = app.output_dir
    if output_dir.exists():
        print(f"✅ Directorio de salida encontrado: {output_dir}")
        csv_files = list(output_dir.glob("*.csv"))
        print(f"📊 Archivos CSV encontrados: {len(csv_files)}")
        for csv in csv_files:
            print(f"   - {csv.name}")
    else:
        print(f"⚠️ Directorio de salida no encontrado: {output_dir}")
    
    print("\n🎉 La aplicación está lista para ejecutarse!")
    print("Para ejecutar: python app.py")
    
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
except Exception as e:
    print(f"❌ Error inesperado: {e}")
