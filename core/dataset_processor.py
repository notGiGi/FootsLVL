# En core/dataset_processor.py (nuevo)
def load_stepup_session(participant_id, trial_conditions):
    """Carga sesión del dataset StepUP-P150"""
    # Procesar archivos .npz
    # Extraer footsteps individuales
    # Convertir a formato Sample(t_ms, left[16], right[16])
    pass

def simulate_nurvv_from_stepup(footstep_data):
    """Simula datos NURVV desde StepUP de alta resolución"""
    # Downsample de 4 sensores/cm² a 16 sensores fijos
    # Mantener timing y características de presión reales
    pass