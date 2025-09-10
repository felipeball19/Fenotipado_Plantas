import os
import pandas as pd
from typing import List, Optional

class DataRepo:
    """Acceso sencillo a la carpeta output/."""
    def __init__(self, base_path: str):
        # base_path debe apuntar al directorio 'output' (ruta absoluta)
        self.base_path = os.path.abspath(base_path)

    def full(self, name: str) -> str:
        """Ruta absoluta a un archivo dentro de output/ (soporta subcarpetas)."""
        return os.path.join(self.base_path, name)

    def list_csvs(self) -> List[str]:
        """Lista de archivos .csv en output/ (nivel raíz)."""
        if not os.path.exists(self.base_path):
            return []
        out = []
        for f in os.listdir(self.base_path):
            if f.lower().endswith(".csv"):
                out.append(f)
        return sorted(out)

    def list_images(self) -> List[str]:
        """Lista de imágenes (png/jpg/jpeg/webp) dentro de output/ (incluye subcarpetas)."""
        if not os.path.exists(self.base_path):
            return []
        exts = (".png", ".jpg", ".jpeg", ".webp")
        out = []
        for root, _, files in os.walk(self.base_path):
            for f in files:
                if f.lower().endswith(exts):
                    rel = os.path.relpath(os.path.join(root, f), self.base_path)
                    out.append(rel)
        return sorted(out)

    def list_all_files(self) -> List[str]:
        """Lista de TODOS los archivos dentro de output/ (con rutas relativas)."""
        if not os.path.exists(self.base_path):
            return []
        out = []
        for root, _, files in os.walk(self.base_path):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), self.base_path)
                out.append(rel)
        return sorted(out)

    def get_csv(self, filename: str, head: bool = False) -> Optional[pd.DataFrame]:
        """Lee un CSV. Si head=True, devuelve solo primeras filas."""
        path = self.full(filename)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        if head:
            return df.head(50)
        return df
