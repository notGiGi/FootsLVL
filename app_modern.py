# app_modern.py
"""
FootLab Premium - Aplicación con UI moderna
Ejecuta con: python app_modern.py
"""

import sys
from PySide6.QtWidgets import QApplication
from ui.modern_ui import ModernFootLabUI

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = ModernFootLabUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
