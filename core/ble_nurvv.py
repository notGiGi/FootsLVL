# core/ble_nurvv.py
"""
Adaptador BLE simple para NURVV que cumple con la interfaz DataSource.
Adapta el sistema avanzado async a la interfaz síncrona que espera la UI.
"""

import asyncio
import threading
import time
import logging
from typing import Callable, Optional, List
from collections import deque
import numpy as np

# Import the advanced BLE system
from sensors.nurvv_ble_advanced import (
    NurvvSystemManager, 
    SynchronizedSample,
    create_nurvv_system
)

logger = logging.getLogger(__name__)

class NurvvBleSource:
    """
    Adaptador simple para integrar NURVV BLE con la UI existente.
    Implementa la misma interfaz que SimulatorSource.
    """
    
    def __init__(self, n_sensors=16, freq=100, auto_connect=True, 
                 sync_window_ms=50.0, **kwargs):
        """
        Args:
            n_sensors: Sensores por pie (16 para NURVV)
            freq: Frecuencia objetivo de muestreo
            auto_connect: Conectar automáticamente al iniciar
            sync_window_ms: Ventana de sincronización entre pies
        """
        self.n_sensors = n_sensors
        self.freq = freq
        self.auto_connect = auto_connect
        self.sync_window_ms = sync_window_ms
        
        # Sistema BLE avanzado
        self.ble_system: Optional[NurvvSystemManager] = None
        
        # Thread y control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._callback: Optional[Callable] = None
        
        # Buffer para muestras
        self._sample_buffer = deque(maxlen=100)
        self._last_sample_time = 0
        
        # Estado de conexión
        self.connected_left = False
        self.connected_right = False
        
        logger.info(f"NurvvBleSource inicializado (n={n_sensors}, freq={freq}Hz)")
    
    def start(self, on_sample: Callable):
        """
        Inicia la fuente de datos BLE.
        Compatible con SimulatorSource.
        """
        if self._running:
            logger.warning("BLE ya está en ejecución")
            return
        
        self._callback = on_sample
        self._running = True
        
        # Crear thread para ejecutar asyncio
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        
        logger.info("NurvvBleSource iniciado")
    
    def stop(self):
        """Detiene la fuente de datos BLE."""
        self._running = False
        
        # Detener loop asyncio
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._shutdown_ble(), self._loop
            )
        
        # Esperar thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        logger.info("NurvvBleSource detenido")
    
    def _run_async(self):
        """Thread principal que ejecuta el loop asyncio."""
        try:
            # Crear nuevo loop para este thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Ejecutar sistema BLE
            self._loop.run_until_complete(self._run_ble_system())
            
        except Exception as e:
            logger.error(f"Error en thread BLE: {e}")
        finally:
            self._loop.close()
            self._loop = None
    
    async def _run_ble_system(self):
        """Ejecuta el sistema BLE avanzado."""
        try:
            # Crear sistema BLE
            self.ble_system = create_nurvv_system(
                sync_window_ms=self.sync_window_ms
            )
            
            # Configurar callback para muestras sincronizadas
            self.ble_system.sync_callback = self._on_synchronized_sample
            
            if self.auto_connect:
                # Descubrir dispositivos
                logger.info("Buscando dispositivos NURVV...")
                devices = await self.ble_system.discover_devices(timeout=10.0)
                
                if len(devices) >= 2:
                    # Conectar automáticamente
                    connected = await self.ble_system.auto_connect(timeout=15.0)
                    
                    if connected:
                        logger.info("✓ Dispositivos NURVV conectados")
                        
                        # Calibrar sincronización temporal
                        await self.ble_system.calibrate_time_sync(duration_seconds=5.0)
                        
                        # Iniciar streaming
                        await self.ble_system.start_synchronized_streaming()
                        
                        # Mantener vivo mientras esté activo
                        while self._running:
                            await asyncio.sleep(0.1)
                            
                            # Emitir muestras del buffer
                            self._emit_buffered_samples()
                    else:
                        logger.error("No se pudo conectar a dispositivos NURVV")
                        # Modo simulador como fallback
                        await self._run_simulator_fallback()
                else:
                    logger.warning(f"Solo {len(devices)} dispositivos encontrados. Usando simulador.")
                    await self._run_simulator_fallback()
            else:
                # Esperar conexión manual
                logger.info("Esperando conexión manual...")
                while self._running and not self._check_connected():
                    await asyncio.sleep(1.0)
                
                if self._check_connected():
                    await self.ble_system.start_synchronized_streaming()
                    
                    while self._running:
                        await asyncio.sleep(0.1)
                        self._emit_buffered_samples()
                        
        except Exception as e:
            logger.error(f"Error en sistema BLE: {e}")
            # Fallback a simulador
            if self._running:
                await self._run_simulator_fallback()
    
    async def _run_simulator_fallback(self):
        """Ejecuta simulador como fallback cuando BLE falla."""
        logger.warning("Ejecutando en modo simulador (BLE no disponible)")
        
        from core.simulator import SimulatorSource
        sim = SimulatorSource(n_sensors=self.n_sensors, freq=self.freq)
        
        # Usar callback del simulador
        def sim_callback(sample):
            if self._callback:
                self._callback(sample)
        
        sim.start(sim_callback)
        
        # Mantener vivo
        while self._running:
            await asyncio.sleep(0.1)
        
        sim.stop()
    
    def _on_synchronized_sample(self, sample: SynchronizedSample):
        """
        Callback cuando llega muestra sincronizada del sistema BLE.
        Convierte al formato esperado por la UI.
        """
        # Convertir a formato de la UI
        ui_sample = {
            "t_ms": sample.timestamp_ms,
            "left": sample.left_pressures[:self.n_sensors],  # Asegurar tamaño
            "right": sample.right_pressures[:self.n_sensors]
        }
        
        # Agregar al buffer
        self._sample_buffer.append(ui_sample)
        
        # Actualizar estado de conexión
        self.connected_left = len(sample.left_pressures) > 0
        self.connected_right = len(sample.right_pressures) > 0
    
    def _emit_buffered_samples(self):
        """Emite muestras del buffer a la frecuencia objetivo."""
        if not self._callback:
            return
        
        current_time = time.time()
        time_since_last = current_time - self._last_sample_time
        
        # Emitir a la frecuencia objetivo
        if time_since_last >= (1.0 / self.freq):
            if self._sample_buffer:
                # Emitir muestra más reciente
                sample = self._sample_buffer[-1]
                self._callback(sample)
                self._last_sample_time = current_time
    
    def _check_connected(self) -> bool:
        """Verifica si hay dispositivos conectados."""
        if not self.ble_system:
            return False
        
        return (self.ble_system.left_client and 
                self.ble_system.left_client.state.value == "connected" and
                self.ble_system.right_client and 
                self.ble_system.right_client.state.value == "connected")
    
    async def _shutdown_ble(self):
        """Cierra conexiones BLE de forma ordenada."""
        if self.ble_system:
            try:
                await self.ble_system.stop_synchronized_streaming()
                await self.ble_system.disconnect_all()
            except Exception as e:
                logger.error(f"Error al cerrar BLE: {e}")
    
    def get_status(self) -> dict:
        """Obtiene estado actual del sistema BLE."""
        return {
            "running": self._running,
            "connected_left": self.connected_left,
            "connected_right": self.connected_right,
            "buffer_size": len(self._sample_buffer),
            "frequency": self.freq
        }

# Alias para compatibilidad
BleSource = NurvvBleSource