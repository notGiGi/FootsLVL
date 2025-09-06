# sensors/nurvv_ble_advanced.py
"""
Advanced BLE integration for NURVV Run insoles with robust connection handling,
data synchronization, and real-time streaming capabilities
"""

import asyncio
import struct
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import json
from pathlib import Path

from bleak import BleakClient, BleakScanner, BleakError
from bleak.backends.characteristic import BleakGATTCharacteristic
import numpy as np

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"

class SensorLocation(Enum):
    LEFT_FOOT = "left"
    RIGHT_FOOT = "right"
    UNKNOWN = "unknown"

@dataclass
class NurvvDeviceInfo:
    """Information about a NURVV device"""
    name: str
    address: str
    rssi: int
    location: SensorLocation
    firmware_version: str = ""
    battery_level: int = 0
    serial_number: str = ""

@dataclass
class PressureFrame:
    """Single frame of pressure data"""
    timestamp_ms: int
    sensor_values: List[float]  # 16 pressure values
    temperature: float = 0.0
    battery_voltage: float = 0.0
    sequence_number: int = 0

@dataclass
class SynchronizedSample:
    """Synchronized sample from both feet"""
    timestamp_ms: int
    left_pressures: List[float]
    right_pressures: List[float]
    sync_quality: float  # 0-1, quality of synchronization
    latency_ms: float

# NURVV BLE Service UUIDs (these would need to be the actual NURVV UUIDs)
class NurvvUUIDs:
    # Main pressure service
    PRESSURE_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
    PRESSURE_CHARACTERISTIC = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
    
    # Device information service
    DEVICE_INFO_SERVICE = "180a"
    FIRMWARE_VERSION_CHAR = "2a26"
    SERIAL_NUMBER_CHAR = "2a25"
    
    # Battery service
    BATTERY_SERVICE = "180f"
    BATTERY_LEVEL_CHAR = "2a19"
    
    # Configuration service (custom)
    CONFIG_SERVICE = "12345678-1234-1234-1234-123456789abc"
    SAMPLING_RATE_CHAR = "12345678-1234-1234-1234-123456789abd"
    CALIBRATION_CHAR = "12345678-1234-1234-1234-123456789abe"

class NurvvDataDecoder:
    """Decodes NURVV BLE data packets"""
    
    def __init__(self):
        self.expected_packet_size = 36  # Expected packet size for 16 sensors
        self.calibration_data = {}
    
    def decode_pressure_packet(self, data: bytes, device_address: str) -> Optional[PressureFrame]:
        """Decode a pressure data packet from NURVV sensor"""
        
        if len(data) < self.expected_packet_size:
            logger.warning(f"Packet too small: {len(data)} bytes")
            return None
        
        try:
            # NURVV packet structure (this would need to match actual protocol):
            # [0:2]   - Packet header
            # [2:4]   - Sequence number  
            # [4:6]   - Timestamp (relative)
            # [6:38]  - 16 pressure values (2 bytes each, little endian)
            # [38:40] - Temperature
            # [40:42] - Battery voltage
            # [42:44] - Checksum
            
            # Unpack header
            header = struct.unpack('<H', data[0:2])[0]
            if header != 0xABCD:  # Example header value
                logger.warning(f"Invalid packet header: {header:04X}")
                return None
            
            # Unpack sequence and timestamp
            sequence = struct.unpack('<H', data[2:4])[0]
            timestamp_relative = struct.unpack('<H', data[4:6])[0]
            
            # Current system timestamp (will be synchronized later)
            timestamp_ms = int(time.time() * 1000)
            
            # Unpack 16 pressure values
            pressure_raw = struct.unpack('<16H', data[6:38])
            
            # Apply calibration if available
            calibration = self.calibration_data.get(device_address, {})
            pressure_values = []
            
            for i, raw_value in enumerate(pressure_raw):
                # Apply offset and scale
                offset = calibration.get(f'offset_{i}', 0)
                scale = calibration.get(f'scale_{i}', 1.0)
                
                # Convert to pressure units (kPa)
                calibrated_value = (raw_value - offset) * scale
                pressure_values.append(max(0.0, calibrated_value))  # Ensure non-negative
            
            # Unpack auxiliary data
            temperature_raw = struct.unpack('<H', data[38:40])[0]
            battery_raw = struct.unpack('<H', data[40:42])[0]
            
            # Convert auxiliary values
            temperature = temperature_raw / 100.0  # Assume temperature in 0.01°C units
            battery_voltage = battery_raw / 1000.0  # Assume voltage in mV
            
            # Verify checksum
            expected_checksum = struct.unpack('<H', data[42:44])[0]
            calculated_checksum = sum(data[0:42]) & 0xFFFF
            
            if expected_checksum != calculated_checksum:
                logger.warning("Checksum mismatch - packet may be corrupted")
                # Continue processing but mark as potentially corrupted
            
            return PressureFrame(
                timestamp_ms=timestamp_ms,
                sensor_values=pressure_values,
                temperature=temperature,
                battery_voltage=battery_voltage,
                sequence_number=sequence
            )
            
        except Exception as e:
            logger.error(f"Error decoding pressure packet: {e}")
            return None
    
    def set_calibration(self, device_address: str, calibration: Dict[str, float]):
        """Set calibration parameters for a device"""
        self.calibration_data[device_address] = calibration

class AdvancedNurvvClient:
    """Advanced NURVV BLE client with robust connection handling"""
    
    def __init__(self, device_info: NurvvDeviceInfo):
        self.device_info = device_info
        self.client: Optional[BleakClient] = None
        self.state = ConnectionState.DISCONNECTED
        
        # Data handling
        self.decoder = NurvvDataDecoder()
        self.data_callback: Optional[Callable[[PressureFrame], None]] = None
        self.frame_buffer = deque(maxlen=1000)  # Buffer recent frames
        
        # Connection parameters
        self.connection_timeout = 10.0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2.0
        self.connection_monitor_interval = 5.0
        
        # Statistics
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'last_packet_time': 0,
            'connection_attempts': 0,
            'last_rssi': 0
        }
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def connect(self) -> bool:
        """Connect to the NURVV device"""
        
        if self.state in [ConnectionState.CONNECTED, ConnectionState.STREAMING]:
            logger.warning(f"Already connected to {self.device_info.name}")
            return True
        
        self.state = ConnectionState.CONNECTING
        self.stats['connection_attempts'] += 1
        
        try:
            # Create BLE client
            self.client = BleakClient(
                self.device_info.address,
                timeout=self.connection_timeout
            )
            
            # Connect with retry logic
            connected = await self._connect_with_retry()
            
            if connected:
                # Setup notifications
                await self._setup_notifications()
                
                # Read device information
                await self._read_device_info()
                
                self.state = ConnectionState.CONNECTED
                logger.info(f"Successfully connected to {self.device_info.name}")
                
                # Start monitoring
                self._start_monitoring()
                
                return True
            else:
                self.state = ConnectionState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to {self.device_info.name}: {e}")
            self.state = ConnectionState.ERROR
            return False
    
    async def _connect_with_retry(self) -> bool:
        """Connect with retry logic"""
        
        for attempt in range(self.max_reconnect_attempts):
            try:
                logger.info(f"Connection attempt {attempt + 1}/{self.max_reconnect_attempts}")
                
                await self.client.connect()
                
                if self.client.is_connected:
                    return True
                    
            except BleakError as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay * (attempt + 1))
            
            except Exception as e:
                logger.error(f"Unexpected error during connection: {e}")
                break
        
        return False
    
    async def _setup_notifications(self):
        """Setup BLE characteristic notifications"""
        
        if not self.client or not self.client.is_connected:
            raise RuntimeError("Client not connected")
        
        # Enable pressure data notifications
        await self.client.start_notify(
            NurvvUUIDs.PRESSURE_CHARACTERISTIC,
            self._pressure_notification_handler
        )
        
        logger.info("Pressure notifications enabled")
    
    async def _read_device_info(self):
        """Read device information characteristics"""
        
        if not self.client or not self.client.is_connected:
            return
        
        try:
            # Read firmware version
            fw_data = await self.client.read_gatt_char(NurvvUUIDs.FIRMWARE_VERSION_CHAR)
            if fw_data:
                self.device_info.firmware_version = fw_data.decode('utf-8').strip()
            
            # Read serial number
            sn_data = await self.client.read_gatt_char(NurvvUUIDs.SERIAL_NUMBER_CHAR)
            if sn_data:
                self.device_info.serial_number = sn_data.decode('utf-8').strip()
            
            # Read battery level
            battery_data = await self.client.read_gatt_char(NurvvUUIDs.BATTERY_LEVEL_CHAR)
            if battery_data:
                self.device_info.battery_level = struct.unpack('B', battery_data)[0]
            
            logger.info(f"Device info - FW: {self.device_info.firmware_version}, "
                       f"SN: {self.device_info.serial_number}, "
                       f"Battery: {self.device_info.battery_level}%")
                       
        except Exception as e:
            logger.warning(f"Could not read all device info: {e}")
    
    def _pressure_notification_handler(self, characteristic: BleakGATTCharacteristic, data: bytes):
        """Handle pressure data notifications"""
        
        try:
            # Update statistics
            self.stats['packets_received'] += 1
            self.stats['last_packet_time'] = time.time()
            
            # Decode the pressure frame
            frame = self.decoder.decode_pressure_packet(data, self.device_info.address)
            
            if frame:
                # Add to buffer
                self.frame_buffer.append(frame)
                
                # Call data callback if set
                if self.data_callback:
                    try:
                        self.data_callback(frame)
                    except Exception as e:
                        logger.error(f"Error in data callback: {e}")
            else:
                self.stats['packets_dropped'] += 1
                
        except Exception as e:
            logger.error(f"Error handling pressure notification: {e}")
            self.stats['packets_dropped'] += 1
    
    def _start_monitoring(self):
        """Start connection monitoring"""
        
        if self._monitoring_task and not self._monitoring_task.done():
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_connection())
    
    async def _monitor_connection(self):
        """Monitor connection health and attempt reconnection if needed"""
        
        while self._running:
            try:
                if not self.client or not self.client.is_connected:
                    logger.warning(f"Connection lost to {self.device_info.name}")
                    self.state = ConnectionState.DISCONNECTED
                    
                    # Attempt reconnection
                    reconnected = await self.connect()
                    if not reconnected:
                        logger.error(f"Failed to reconnect to {self.device_info.name}")
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                
                # Check data flow
                current_time = time.time()
                time_since_last_packet = current_time - self.stats['last_packet_time']
                
                if time_since_last_packet > 5.0:  # No data for 5 seconds
                    logger.warning(f"No data received from {self.device_info.name} for {time_since_last_packet:.1f}s")
                
                # Read RSSI periodically
                try:
                    rssi = await self.client.get_rssi()
                    self.stats['last_rssi'] = rssi
                    
                    if rssi < -80:  # Weak signal
                        logger.warning(f"Weak signal from {self.device_info.name}: {rssi} dBm")
                        
                except Exception:
                    pass  # RSSI reading not critical
                
                await asyncio.sleep(self.connection_monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(self.connection_monitor_interval)
    
    async def start_streaming(self, data_callback: Callable[[PressureFrame], None]):
        """Start streaming pressure data"""
        
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError("Must be connected before starting stream")
        
        self.data_callback = data_callback
        self.state = ConnectionState.STREAMING
        
        logger.info(f"Started streaming from {self.device_info.name}")
    
    async def stop_streaming(self):
        """Stop streaming pressure data"""
        
        self.data_callback = None
        
        if self.state == ConnectionState.STREAMING:
            self.state = ConnectionState.CONNECTED
        
        logger.info(f"Stopped streaming from {self.device_info.name}")
    
    async def disconnect(self):
        """Disconnect from the device"""
        
        self._running = False
        
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect BLE client
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(NurvvUUIDs.PRESSURE_CHARACTERISTIC)
                await self.client.disconnect()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        self.state = ConnectionState.DISCONNECTED
        logger.info(f"Disconnected from {self.device_info.name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection and data statistics"""
        
        stats = self.stats.copy()
        stats.update({
            'device_name': self.device_info.name,
            'device_address': self.device_info.address,
            'connection_state': self.state.value,
            'firmware_version': self.device_info.firmware_version,
            'battery_level': self.device_info.battery_level,
            'buffer_size': len(self.frame_buffer),
            'packet_loss_rate': (self.stats['packets_dropped'] / max(1, self.stats['packets_received'])) * 100
        })
        
        return stats

class NurvvSystemManager:
    """Manages dual NURVV sensors with synchronization"""
    
    def __init__(self, sync_window_ms: float = 50.0):
        self.left_client: Optional[AdvancedNurvvClient] = None
        self.right_client: Optional[AdvancedNurvvClient] = None
        
        # Synchronization
        self.sync_window_ms = sync_window_ms
        self.left_buffer = deque(maxlen=200)
        self.right_buffer = deque(maxlen=200)
        self.sync_callback: Optional[Callable[[SynchronizedSample], None]] = None
        
        # Timing synchronization
        self.time_offset_left = 0.0
        self.time_offset_right = 0.0
        
        # Statistics
        self.sync_stats = {
            'synchronized_samples': 0,
            'dropped_samples': 0,
            'average_latency': 0.0,
            'sync_quality_history': deque(maxlen=100)
        }
        
        self._synchronizer_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def discover_devices(self, timeout: float = 10.0) -> List[NurvvDeviceInfo]:
        """Discover NURVV devices"""
        
        logger.info("Scanning for NURVV devices...")
        discovered_devices = []
        
        try:
            # Scan for BLE devices
            devices = await BleakScanner.discover(timeout=timeout)
            
            for device in devices:
                # Filter for NURVV devices (by name or service UUIDs)
                if self._is_nurvv_device(device):
                    device_info = await self._identify_nurvv_device(device)
                    if device_info:
                        discovered_devices.append(device_info)
            
            logger.info(f"Found {len(discovered_devices)} NURVV devices")
            
        except Exception as e:
            logger.error(f"Error during device discovery: {e}")
        
        return discovered_devices
    
    def _is_nurvv_device(self, device) -> bool:
        """Check if a BLE device is a NURVV sensor"""
        
        # Check device name
        if device.name and "nurvv" in device.name.lower():
            return True
        
        # Check advertised services
        if device.metadata and 'uuids' in device.metadata:
            advertised_uuids = [uuid.lower() for uuid in device.metadata['uuids']]
            if NurvvUUIDs.PRESSURE_SERVICE.lower() in advertised_uuids:
                return True
        
        return False
    
    async def _identify_nurvv_device(self, device) -> Optional[NurvvDeviceInfo]:
        """Identify NURVV device location (left/right foot)"""
        
        try:
            # Try to connect temporarily to read device info
            temp_client = BleakClient(device.address, timeout=5.0)
            
            try:
                await temp_client.connect()
                
                # Read device-specific identifier or configuration
                # This would depend on how NURVV devices identify themselves
                location = SensorLocation.UNKNOWN
                
                # Example: read a custom characteristic that indicates foot location
                try:
                    config_data = await temp_client.read_gatt_char(NurvvUUIDs.CALIBRATION_CHAR)
                    if config_data and len(config_data) > 0:
                        location_byte = config_data[0]
                        if location_byte == 0x01:
                            location = SensorLocation.LEFT_FOOT
                        elif location_byte == 0x02:
                            location = SensorLocation.RIGHT_FOOT
                except Exception:
                    pass
                
                # If unable to determine from characteristics, use device name or address
                if location == SensorLocation.UNKNOWN:
                    device_name = device.name or ""
                    if "left" in device_name.lower() or "l" in device_name.lower():
                        location = SensorLocation.LEFT_FOOT
                    elif "right" in device_name.lower() or "r" in device_name.lower():
                        location = SensorLocation.RIGHT_FOOT
                    else:
                        # Use MAC address pattern as fallback
                        # Odd last byte = left, even = right (example heuristic)
                        last_byte = int(device.address.split(':')[-1], 16)
                        location = SensorLocation.LEFT_FOOT if last_byte % 2 == 1 else SensorLocation.RIGHT_FOOT
                
                await temp_client.disconnect()
                
                return NurvvDeviceInfo(
                    name=device.name or f"NURVV-{device.address[-5:]}",
                    address=device.address,
                    rssi=device.rssi or 0,
                    location=location
                )
                
            except Exception as e:
                logger.warning(f"Could not identify device {device.address}: {e}")
                if temp_client.is_connected:
                    await temp_client.disconnect()
                
        except Exception as e:
            logger.error(f"Error identifying NURVV device: {e}")
        
        return None
    
    async def connect_devices(self, left_device: NurvvDeviceInfo, 
                            right_device: NurvvDeviceInfo) -> bool:
        """Connect to both left and right foot devices"""
        
        # Create clients
        self.left_client = AdvancedNurvvClient(left_device)
        self.right_client = AdvancedNurvvClient(right_device)
        
        # Connect both devices
        left_connected = await self.left_client.connect()
        right_connected = await self.right_client.connect()
        
        if left_connected and right_connected:
            logger.info("Successfully connected to both NURVV devices")
            
            # Setup data callbacks for synchronization
            await self.left_client.start_streaming(self._left_data_callback)
            await self.right_client.start_streaming(self._right_data_callback)
            
            # Start synchronizer
            self._start_synchronizer()
            
            return True
        else:
            logger.error("Failed to connect to one or both devices")
            
            # Clean up partial connections
            if left_connected:
                await self.left_client.disconnect()
            if right_connected:
                await self.right_client.disconnect()
            
            return False
    
    def _left_data_callback(self, frame: PressureFrame):
        """Callback for left foot data"""
        
        # Apply time offset correction
        corrected_frame = PressureFrame(
            timestamp_ms=frame.timestamp_ms + self.time_offset_left,
            sensor_values=frame.sensor_values,
            temperature=frame.temperature,
            battery_voltage=frame.battery_voltage,
            sequence_number=frame.sequence_number
        )
        
        self.left_buffer.append(corrected_frame)
    
    def _right_data_callback(self, frame: PressureFrame):
        """Callback for right foot data"""
        
        # Apply time offset correction
        corrected_frame = PressureFrame(
            timestamp_ms=frame.timestamp_ms + self.time_offset_right,
            sensor_values=frame.sensor_values,
            temperature=frame.temperature,
            battery_voltage=frame.battery_voltage,
            sequence_number=frame.sequence_number
        )
        
        self.right_buffer.append(corrected_frame)
    
    def _start_synchronizer(self):
        """Start the data synchronization task"""
        
        self._running = True
        self._synchronizer_task = asyncio.create_task(self._synchronize_data())
    
    async def _synchronize_data(self):
        """Synchronize data from both feet"""
        
        while self._running:
            try:
                # Try to find matching frames
                if len(self.left_buffer) > 0 and len(self.right_buffer) > 0:
                    synchronized = self._find_synchronized_frames()
                    
                    if synchronized and self.sync_callback:
                        try:
                            self.sync_callback(synchronized)
                            self.sync_stats['synchronized_samples'] += 1
                        except Exception as e:
                            logger.error(f"Error in sync callback: {e}")
                
                await asyncio.sleep(0.01)  # 10ms sync loop
                
            except Exception as e:
                logger.error(f"Error in data synchronization: {e}")
                await asyncio.sleep(0.1)
    
    def _find_synchronized_frames(self) -> Optional[SynchronizedSample]:
        """Find temporally synchronized frames from both feet"""
        
        if not self.left_buffer or not self.right_buffer:
            return None
        
        # Find the closest temporal match
        left_frame = None
        right_frame = None
        best_time_diff = float('inf')
        best_left_idx = -1
        best_right_idx = -1
        
        # Search within sync window
        for left_idx, left_candidate in enumerate(self.left_buffer):
            for right_idx, right_candidate in enumerate(self.right_buffer):
                time_diff = abs(left_candidate.timestamp_ms - right_candidate.timestamp_ms)
                
                if time_diff <= self.sync_window_ms and time_diff < best_time_diff:
                    best_time_diff = time_diff
                    left_frame = left_candidate
                    right_frame = right_candidate
                    best_left_idx = left_idx
                    best_right_idx = right_idx
        
        if left_frame and right_frame:
            # Remove synchronized frames from buffers
            if best_left_idx >= 0:
                for _ in range(best_left_idx + 1):
                    self.left_buffer.popleft()
            
            if best_right_idx >= 0:
                for _ in range(best_right_idx + 1):
                    self.right_buffer.popleft()
            
            # Calculate sync quality
            sync_quality = max(0.0, 1.0 - (best_time_diff / self.sync_window_ms))
            self.sync_stats['sync_quality_history'].append(sync_quality)
            
            # Use average timestamp
            avg_timestamp = int((left_frame.timestamp_ms + right_frame.timestamp_ms) / 2)
            
            return SynchronizedSample(
                timestamp_ms=avg_timestamp,
                left_pressures=left_frame.sensor_values,
                right_pressures=right_frame.sensor_values,
                sync_quality=sync_quality,
                latency_ms=best_time_diff
            )
        
        return None
    
    def start_synchronized_streaming(self, callback: Callable[[SynchronizedSample], None]):
        """Start synchronized streaming from both devices"""
        
        self.sync_callback = callback
        logger.info("Started synchronized streaming")
    
    def stop_synchronized_streaming(self):
        """Stop synchronized streaming"""
        
        self.sync_callback = None
        logger.info("Stopped synchronized streaming")
    
    async def disconnect_all(self):
        """Disconnect from all devices"""
        
        self._running = False
        
        # Cancel synchronizer
        if self._synchronizer_task and not self._synchronizer_task.done():
            self._synchronizer_task.cancel()
            try:
                await self._synchronizer_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect devices
        if self.left_client:
            await self.left_client.disconnect()
        
        if self.right_client:
            await self.right_client.disconnect()
        
        logger.info("Disconnected from all NURVV devices")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = {
            'synchronization': self.sync_stats.copy(),
            'left_device': self.left_client.get_statistics() if self.left_client else None,
            'right_device': self.right_client.get_statistics() if self.right_client else None,
            'buffer_sizes': {
                'left': len(self.left_buffer),
                'right': len(self.right_buffer)
            }
        }
        
        # Calculate average sync quality
        if self.sync_stats['sync_quality_history']:
            stats['synchronization']['average_sync_quality'] = np.mean(
                list(self.sync_stats['sync_quality_history']))
        
        return stats
    
    async def calibrate_time_sync(self, duration_seconds: float = 10.0):
        """Calibrate time synchronization between devices"""
        
        logger.info(f"Starting time synchronization calibration for {duration_seconds}s")
        
        # Collect timestamp pairs for synchronization
        sync_pairs = []
        start_time = time.time()
        
        def collect_sync_data(sample: SynchronizedSample):
            nonlocal sync_pairs
            sync_pairs.append({
                'timestamp': sample.timestamp_ms,
                'latency': sample.latency_ms,
                'quality': sample.sync_quality
            })
        
        # Temporarily set sync callback
        old_callback = self.sync_callback
        self.sync_callback = collect_sync_data
        
        # Wait for calibration period
        await asyncio.sleep(duration_seconds)
        
        # Restore original callback
        self.sync_callback = old_callback
        
        # Analyze collected data
        if len(sync_pairs) > 10:
            latencies = [p['latency'] for p in sync_pairs]
            qualities = [p['quality'] for p in sync_pairs]
            
            avg_latency = np.mean(latencies)
            avg_quality = np.mean(qualities)
            
            # Simple time offset correction
            # In practice, this would be more sophisticated
            self.time_offset_left = -avg_latency / 2
            self.time_offset_right = avg_latency / 2
            
            logger.info(f"Time sync calibration complete:")
            logger.info(f"  Average latency: {avg_latency:.1f}ms")
            logger.info(f"  Average quality: {avg_quality:.3f}")
            logger.info(f"  Applied offsets: L={self.time_offset_left:.1f}ms, R={self.time_offset_right:.1f}ms")
            
            return True
        else:
            logger.warning("Insufficient data for time synchronization calibration")
            return False

# Factory function and utilities
def create_nurvv_system(sync_window_ms: float = 50.0) -> NurvvSystemManager:
    """Create a NURVV system manager"""
    return NurvvSystemManager(sync_window_ms=sync_window_ms)

async def auto_connect_nurvv_system() -> Optional[NurvvSystemManager]:
    """Automatically discover and connect to NURVV devices"""
    
    system = create_nurvv_system()
    
    try:
        # Discover devices
        devices = await system.discover_devices(timeout=15.0)
        
        if len(devices) < 2:
            logger.error(f"Found only {len(devices)} devices, need 2 for dual foot setup")
            return None
        
        # Find left and right devices
        left_device = None
        right_device = None
        
        for device in devices:
            if device.location == SensorLocation.LEFT_FOOT:
                left_device = device
            elif device.location == SensorLocation.RIGHT_FOOT:
                right_device = device
        
        if not left_device or not right_device:
            logger.error("Could not identify left and right foot devices")
            return None
        
        # Connect to devices
        connected = await system.connect_devices(left_device, right_device)
        
        if connected:
            # Calibrate time synchronization
            await system.calibrate_time_sync(duration_seconds=5.0)
            
            logger.info("NURVV system ready for use")
            return system
        else:
            logger.error("Failed to connect to NURVV devices")
            return None
            
    except Exception as e:
        logger.error(f"Error setting up NURVV system: {e}")
        return None

# Example usage
async def example_usage():
    """Example of how to use the NURVV system"""
    
    # Auto-connect to devices
    nurvv_system = await auto_connect_nurvv_system()
    
    if not nurvv_system:
        print("Failed to setup NURVV system")
        return
    
    # Define data callback
    def handle_synchronized_data(sample: SynchronizedSample):
        print(f"Synchronized sample at {sample.timestamp_ms}ms:")
        print(f"  Left pressures: {sample.left_pressures[:5]}... (showing first 5)")
        print(f"  Right pressures: {sample.right_pressures[:5]}... (showing first 5)")
        print(f"  Sync quality: {sample.sync_quality:.3f}")
        print(f"  Latency: {sample.latency_ms:.1f}ms")
        print()
    
    # Start streaming
    nurvv_system.start_synchronized_streaming(handle_synchronized_data)
    
    # Stream for 30 seconds
    await asyncio.sleep(30)
    
    # Get statistics
    stats = nurvv_system.get_system_statistics()
    print("System Statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Disconnect
    await nurvv_system.disconnect_all()

if __name__ == "__main__":
    asyncio.run(example_usage())