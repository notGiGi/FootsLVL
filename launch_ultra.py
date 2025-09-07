# launch_ultra.py
"""
Launcher para la UI Ultra Premium
"""

import sys
from PySide6.QtWidgets import QApplication

# Importar después de crear QApplication para evitar warnings
app = QApplication(sys.argv)
app.setStyle("Fusion")

from ui.ultra_premium_ui import UltraPremiumFootLabUI

window = UltraPremiumFootLabUI()
window.show()

sys.exit(app.exec())
