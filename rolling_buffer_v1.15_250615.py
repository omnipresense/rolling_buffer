# need to 'pip install pyserial matplotlib numpy' to install required libraries
import serial
import time
import json
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to prevent hanging
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional

# Configure logging - simplified format
logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')
logger = logging.getLogger(__name__)

class RadarProcessor:
    def __init__(self, port: str = "COM4", baudrate: int = 19200):
        """Initialize radar processor with serial port configuration."""
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        
        # Signal processing constants (SAMPLE_RATE will be updated from sensor)
        self.WINDOW_SIZE = 128
        self.FFT_SIZE = 4096
        self.NUM_BLOCKS = 32
        self.SAMPLE_RATE = 10000  # Default value, will be updated from sensor
        
        # Speed conversion calculation will be updated after getting sample rate
        self.SPEED_CONVERSION_FACTOR = None  # Will be calculated after sample rate is known
        self.SPEED_RESOLUTION = None  # Will be read from sensor
        self.MAGNITUDE_THRESHOLD = 20
        self.MAX_PEAKS = 5
        
        # Pre-compute Hanning window
        self.hanning_window = np.hanning(self.WINDOW_SIZE)
    
    def connect(self) -> bool:
        """Establish connection to radar sensor."""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1,
                writeTimeout=2
            )
            
            # Clear buffers
            self.serial_port.reset_output_buffer()
            self.serial_port.reset_input_buffer()
            
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            logger.info("Disconnected from radar sensor")
    
    def send_command(self, command: str, match_criteria: str = '{') -> Optional[List[bytes]]:
        """Send command to radar sensor and wait for response."""
        if not self.serial_port:
            logger.error("Serial port not connected")
            return None
            
        try:
            # Send command
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_port.write(command_bytes)
            
            # Wait for response
            max_attempts = 10
            attempt = 0
            
            while attempt < max_attempts:
                response = self.serial_port.readlines()
                if response:
                    response_str = str(response)
                    if match_criteria in response_str:
                        return response
                attempt += 1
                time.sleep(0.1)
            
            logger.warning(f"No valid response received for command: {command}")
            return None
            
        except serial.SerialException as e:
            logger.error(f"Serial communication error: {e}")
            return None
    
    def get_sensor_parameters(self) -> bool:
        """Get sensor parameters from ?? command and update sample rate and speed resolution."""
        
        # Send ?? command to get sensor information
        response = self.send_command('??')
        if not response:
            logger.error("Failed to get sensor parameters")
            return False
        
        try:
            # Decode response messages
            messages = []
            for msg in response:
                decoded = msg.decode('utf-8').strip()
                if decoded:  # Skip empty lines
                    messages.append(decoded)
            
            # Parse each line looking for SamplingRate, Resolution, and MagnitudeMin
            sampling_rate_found = False
            resolution_found = False
            magnitude_min_found = False
            
            for msg in messages:
                try:
                    if msg.startswith('{'):
                        parsed = json.loads(msg)
                        
                        # Look for SamplingRate
                        if 'SamplingRate' in parsed:
                            self.SAMPLE_RATE = int(parsed['SamplingRate'])
                            sampling_rate_found = True
                        
                        # Look for Resolution
                        if 'Resolution' in parsed:
                            self.SPEED_RESOLUTION = float(parsed['Resolution'])
                            resolution_found = True
                        
                        # Look for MagnitudeMin
                        if 'MagnitudeMin' in parsed:
                            self.MAGNITUDE_THRESHOLD = int(parsed['MagnitudeMin'])
                            magnitude_min_found = True
                            
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            if not sampling_rate_found:
                logger.warning("Could not find SamplingRate in sensor response, using default 10000")
                self.SAMPLE_RATE = 10000
            
            if not resolution_found:
                logger.warning("Could not find Resolution in sensor response")
                self.SPEED_RESOLUTION = None
            
            if not magnitude_min_found:
                logger.warning("Could not find MagnitudeMin in sensor response, using default 20")
            
            # Calculate speed conversion factor now that we have sample rate
            frequency_per_bin = self.SAMPLE_RATE / self.FFT_SIZE  # Hz per bin
            self.SPEED_CONVERSION_FACTOR = 0.0063 * frequency_per_bin  # (m/s) per bin
            
            # Report the parameters
            logger.info(f"Sample Rate: {self.SAMPLE_RATE} Hz")
            
            if self.SPEED_RESOLUTION is not None:
                logger.info(f"Speed Resolution: {self.SPEED_RESOLUTION} m/s")
            else:
                logger.info("Speed Resolution: Not available from sensor")
            
            logger.info(f"Hz/bin: {frequency_per_bin:.4f} Hz")
            logger.info(f"Magnitude Threshold: {self.MAGNITUDE_THRESHOLD}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing sensor parameters: {e}")
            logger.warning("Using default sample rate: 10000 Hz")
            self.SAMPLE_RATE = 10000
            frequency_per_bin = self.SAMPLE_RATE / self.FFT_SIZE
            self.SPEED_CONVERSION_FACTOR = 0.0063 * frequency_per_bin
            logger.info(f"Sample Rate: {self.SAMPLE_RATE} Hz")
            logger.info("Speed Resolution: Not available from sensor")
            logger.info(f"Hz/bin: {frequency_per_bin:.4f} Hz")
            logger.info(f"Magnitude Threshold: {self.MAGNITUDE_THRESHOLD}")
            return False
    
    def initialize_sensor(self) -> bool:
        """Initialize sensor to proper mode."""
        # First get sensor parameters including sample rate
        if not self.get_sensor_parameters():
            logger.warning("Failed to get sensor parameters, continuing with defaults")
        
        # Deactivate any active mode
        if not self.send_command('PI'):
            return False
        
        # Set peak detect feature
        if not self.send_command('K+'):
            return False
            
        # Set to G1 mode for large data capture
        if not self.send_command('G1'):
            return False
            
        logger.info("Sensor initialized successfully")
        return True
    
    def find_peaks(self, magnitude_array: np.ndarray) -> List[Tuple[int, float]]:
        """Find peak bins in magnitude spectrum and return (bin, signed_speed) pairs."""
        peaks = []
        
        # For a 4096-point FFT:
        # - Bins 0 to 2047: positive frequencies (positive speeds)
        # - Bins 2048 to 4095: negative frequencies (negative speeds)
        # - Bin 2048 corresponds to -sample_rate/2, bin 4095 corresponds to -sample_rate/4096
        
        # Search for peaks in positive frequency range (bins 1 to 2047)
        for bin_idx in range(1, self.FFT_SIZE // 2):
            left_val = magnitude_array[bin_idx - 1]
            center_val = magnitude_array[bin_idx]
            right_val = magnitude_array[bin_idx + 1] if bin_idx < self.FFT_SIZE // 2 - 1 else 0
            
            # Check if this is a local maximum above threshold
            if (center_val > left_val and 
                center_val > right_val and 
                center_val > self.MAGNITUDE_THRESHOLD):
                
                # Calculate corresponding negative frequency bin
                negative_bin = self.FFT_SIZE - bin_idx
                negative_magnitude = magnitude_array[negative_bin]
                
                # Determine which direction has higher magnitude
                if center_val >= negative_magnitude:
                    # Positive speed (target moving away)
                    speed = bin_idx * self.SPEED_CONVERSION_FACTOR
                    peaks.append((bin_idx, speed))
                    logger.debug(f"Positive peak: bin {bin_idx} (mag {center_val:.1f}) > bin {negative_bin} (mag {negative_magnitude:.1f}) → speed +{speed:.2f}")
                else:
                    # Negative speed (target moving toward)
                    speed = -bin_idx * self.SPEED_CONVERSION_FACTOR
                    peaks.append((negative_bin, speed))
                    logger.debug(f"Negative peak: bin {negative_bin} (mag {negative_magnitude:.1f}) > bin {bin_idx} (mag {center_val:.1f}) → speed {speed:.2f}")
        
        # Also check peaks in the negative frequency range that don't have positive counterparts
        for bin_idx in range(self.FFT_SIZE // 2 + 1, self.FFT_SIZE - 1):
            left_val = magnitude_array[bin_idx - 1]
            center_val = magnitude_array[bin_idx]
            right_val = magnitude_array[bin_idx + 1]
            
            if (center_val > left_val and 
                center_val > right_val and 
                center_val > self.MAGNITUDE_THRESHOLD):
                
                # Calculate corresponding positive frequency bin
                positive_bin = self.FFT_SIZE - bin_idx
                
                # Only add if we haven't already processed this frequency pair
                if positive_bin >= self.FFT_SIZE // 2 or magnitude_array[positive_bin] <= self.MAGNITUDE_THRESHOLD:
                    # This negative frequency peak doesn't have a significant positive counterpart
                    frequency_bin = bin_idx - self.FFT_SIZE  # Convert to negative frequency index
                    speed = frequency_bin * self.SPEED_CONVERSION_FACTOR
                    peaks.append((bin_idx, speed))
                    logger.debug(f"Standalone negative peak: bin {bin_idx} → speed {speed:.2f}")
        
        return peaks
    
    def process_iq_block(self, i_data: List[float], q_data: List[float], block_idx: int = 0, sample_time: float = 0.0) -> List[float]:
        """Process a single block of I/Q data and return detected speeds."""
        # Convert to numpy arrays
        i_array = np.array(i_data, dtype=float)
        q_array = np.array(q_data, dtype=float)
        
        # Remove DC component
        i_mean = np.mean(i_array)
        q_mean = np.mean(q_array)
        i_array = (i_array - i_mean) * 3.3 / 4096
        q_array = (q_array - q_mean) * 3.3 / 4096
        
        # Apply Hanning window
        i_array *= self.hanning_window
        q_array *= self.hanning_window
        
        # Create complex signal and perform FFT
        complex_signal = i_array + 1j * q_array
        fft_result = np.fft.fft(complex_signal, self.FFT_SIZE)
        magnitude_spectrum = np.abs(fft_result)
        
        # Find peaks (returns list of (bin_idx, signed_speed) tuples)
        peak_data = self.find_peaks(magnitude_spectrum)
        
        # Calculate block time
        block_time = sample_time + block_idx * (128 / self.SAMPLE_RATE)
        
        # Sort peaks by magnitude (highest first) for JSON output
        if peak_data:
            # Create list of (bin_idx, speed, magnitude) tuples
            peak_magnitudes = [(bin_idx, speed, magnitude_spectrum[bin_idx]) for bin_idx, speed in peak_data]
            # Sort by magnitude in descending order (highest first)
            peak_magnitudes.sort(key=lambda x: x[2], reverse=True)
            
            # Extract speeds and magnitudes in sorted order
            speeds_list = [round(speed, 2) for bin_idx, speed, magnitude in peak_magnitudes]
            magnitudes_list = [round(magnitude) for bin_idx, speed, magnitude in peak_magnitudes]  # No decimal places
            
            print(json.dumps({
                "Time": round(block_time, 3), 
                "Block": block_idx, 
                "Magnitude": magnitudes_list,
                "Speeds": speeds_list
            }))
        else:
            print(json.dumps({
                "Time": round(block_time, 3), 
                "Block": block_idx, 
                "Magnitude": [],
                "Speeds": []
            }))
        
        if not peak_data:
            return []
        
        # Sort by magnitude (need to get magnitude from spectrum) and take top peaks
        peak_magnitudes = [(bin_idx, speed, magnitude_spectrum[bin_idx]) for bin_idx, speed in peak_data]
        peak_magnitudes.sort(key=lambda x: x[2], reverse=True)  # Sort by magnitude
        top_peaks = peak_magnitudes[:self.MAX_PEAKS]
        
        # Extract the signed speeds
        speeds = [speed for bin_idx, speed, magnitude in top_peaks]
        
        return speeds
    
    def capture_and_process(self) -> Dict[float, List[float]]:
        """Capture data from sensor and process all blocks."""
        # Trigger data capture
        response = self.send_command('S!')
        if not response:
            logger.error("No response received from sensor")
            return {}
        
        # Removed: "Received X response lines" logging
        
        try:
            # Decode response messages and filter out empty lines
            messages = []
            for i, msg in enumerate(response):
                decoded = msg.decode('utf-8').strip()
                if decoded:  # Skip empty lines
                    messages.append(decoded)
                    logger.debug(f"Response line {len(messages)-1}: {decoded[:100]}...")
            
            # Removed: "Valid JSON lines: X" logging
            
            # Based on actual sensor output format, we expect:
            # Line 0: {"sample_time" : "964.003"}
            # Line 1: {"trigger_time" : "964.105"} 
            # Line 2: {"I":[...]} - 4096 samples
            # Line 3: {"Q":[...]} - 4096 samples
            
            if len(messages) < 4:
                logger.error(f"Expected at least 4 valid JSON lines, got {len(messages)}")
                logger.info("Valid response lines received:")
                for i, msg in enumerate(messages):
                    logger.info(f"  Line {i}: {msg[:200]}...")
                return {}
            
            # Parse the structured response
            try:
                # Parse sample time (line 0)
                sample_data = json.loads(messages[0])
                sample_time = float(sample_data["sample_time"])
                logger.info(f"1st Sample Time: {sample_time}")
                
                # Parse trigger time (line 1)
                trigger_data = json.loads(messages[1])
                trigger_time = float(trigger_data["trigger_time"])
                logger.info(f"Trigger time: {trigger_time}")
                
                # Parse I data (line 2)
                i_json = json.loads(messages[2])
                i_data = i_json["I"]
                # Removed: "I data length: X" logging
                
                # Parse Q data (line 3)
                q_json = json.loads(messages[3])
                q_data = q_json["Q"]
                # Removed: "Q data length: X" logging
                
            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                logger.error(f"Error parsing structured JSON response: {e}")
                # Fallback: search through all lines for the data
                sample_data = None
                trigger_data = None
                i_data = None
                q_data = None
                
                for i, msg in enumerate(messages):
                    try:
                        parsed = json.loads(msg)
                        if 'sample_time' in parsed:
                            sample_data = parsed
                            logger.info(f"Found sample data at line {i}")
                        elif 'trigger_time' in parsed:
                            trigger_data = parsed
                            logger.info(f"Found trigger data at line {i}")
                        elif 'I' in parsed:
                            i_data = parsed["I"]
                            logger.info(f"Found I data at line {i}, length: {len(i_data)}")
                        elif 'Q' in parsed:
                            q_data = parsed["Q"]
                            logger.info(f"Found Q data at line {i}, length: {len(q_data)}")
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                if not sample_data:
                    logger.warning("Could not find sample_time data in response")
                    sample_time = 0.0  # Default value
                else:
                    sample_time = float(sample_data["sample_time"])
                    logger.info(f"1st Sample Time: {sample_time}")
                
                if not trigger_data:
                    logger.error("Could not find trigger_time data in response")
                    return {}
                
                if not i_data or not q_data:
                    logger.error("Could not find I/Q data in response")
                    return {}
                
                trigger_time = float(trigger_data["trigger_time"])
            
            # Validate data lengths
            expected_length = self.NUM_BLOCKS * self.WINDOW_SIZE  # 32 * 128 = 4096
            if len(i_data) != expected_length or len(q_data) != expected_length:
                logger.warning(f"Data length mismatch: I={len(i_data)}, Q={len(q_data)}, expected={expected_length}")
            
            logger.info(f"Processing {self.NUM_BLOCKS} blocks of data, trigger_time={trigger_time}")
            
            # Output raw data samples for debugging
            # logger.info("="*50)
            # logger.info("FULL RAW DATA CAPTURE:")
            # logger.info("="*50)
            # logger.info(f"I data (all {len(i_data)} samples): {i_data}")
            # logger.info(f"Q data (all {len(q_data)} samples): {q_data}")
            # logger.info("="*50)
            
            results = {}
            total_detections = 0
            
            # Process each block
            for block_idx in range(self.NUM_BLOCKS):
                start_idx = self.WINDOW_SIZE * block_idx
                end_idx = self.WINDOW_SIZE * (block_idx + 1)
                
                block_i = i_data[start_idx:end_idx]
                block_q = q_data[start_idx:end_idx]
                
                logger.debug(f"Processing block {block_idx}: samples {start_idx}-{end_idx}")
                
                speeds = self.process_iq_block(block_i, block_q, block_idx, sample_time)
                
                if speeds:  # Only store if we found speeds
                    timestamp = trigger_time + block_idx * (self.WINDOW_SIZE / self.SAMPLE_RATE)
                    results[timestamp] = speeds
                    total_detections += len(speeds)
                    logger.debug(f"Block {block_idx}: found {len(speeds)} speeds: {speeds}")
                else:
                    logger.debug(f"Block {block_idx}: no speeds detected")
            
            logger.info(f"Processing complete: {len(results)} blocks with detections")
            # Removed: "Total speed detections: X" logging
            
            # Reset sensor back to rolling buffer mode after data capture
            try:
                logger.debug("Sending G1 command to reset sensor to rolling buffer mode...")
                self.send_command('G1')
                logger.debug("G1 command sent successfully")
            except Exception as reset_error:
                logger.warning(f"Failed to send G1 reset command: {reset_error}")
            
            if not results:
                logger.warning("No speeds detected in any block. This could indicate:")
                logger.warning("  1. No signal present above threshold")
                logger.warning("  2. Threshold too high (current: {})".format(self.MAGNITUDE_THRESHOLD))
                logger.warning("  3. Issue with FFT processing")
                logger.warning("  Try lowering the magnitude threshold or check signal strength")
            
            return results
            
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing sensor data: {e}")
            logger.info("Full response dump:")
            for i, msg in enumerate(messages if 'messages' in locals() else [str(r) for r in response]):
                logger.info(f"  Response {i}: {msg}")
            return {}
    
    def plot_results(self, results: Dict[float, List[float]]):
        """Plot the detected speeds over time, showing positive and negative speeds."""
        if not results:
            logger.warning("No data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        all_times = []
        all_speeds = []
        positive_times = []
        positive_speeds = []
        negative_times = []
        negative_speeds = []
        
        for timestamp, speeds in results.items():
            for speed in speeds:
                all_times.append(timestamp)
                all_speeds.append(speed)
                
                if speed > 0:
                    positive_times.append(timestamp)
                    positive_speeds.append(speed)
                elif speed < 0:
                    negative_times.append(timestamp)
                    negative_speeds.append(speed)
        
        if all_times:
            # Plot positive speeds in blue, negative in red
            if positive_times:
                plt.scatter(positive_times, positive_speeds, alpha=0.7, s=40, c='blue', 
                          label='Moving Away (+)', marker='^')
            if negative_times:
                plt.scatter(negative_times, negative_speeds, alpha=0.7, s=40, c='red', 
                          label='Moving Toward (-)', marker='v')
            
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero Speed')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Speed (m/s)')
            plt.title('Detected Radar Targets (Doppler Speed vs Time)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add some statistics
            max_pos = max(positive_speeds) if positive_speeds else 0
            max_neg = min(negative_speeds) if negative_speeds else 0
            
            # Save plot as PNG file
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"radar_plot_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved as: {filename}")
            
            # Use non-blocking show to prevent hanging
            plt.show(block=False)
            plt.pause(0.1)  # Brief pause to ensure plot displays
            # Removed: "Plot displayed. Close plot window to continue..." logging
        else:
            logger.info("No speeds detected to plot")
    
    def reset_sensor(self):
        """Reset sensor back to normal mode."""
        self.send_command('PI')
        logger.info("Sensor reset to normal mode")


def main():
    """Main execution function."""
    processor = RadarProcessor(port="COM4")
    
    try:
        # Connect and initialize
        if not processor.connect():
            return
        
        if not processor.initialize_sensor():
            return
        
        # Interactive loop
        logger.info("Radar processor ready. Commands:")
        logger.info("  'Trig' - capture and process data")
        logger.info("  'quit' - exit")
        
        while True:
            user_input = input("Enter command: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input == 'Trig':
                logger.info("Triggering data capture...")
                results = processor.capture_and_process()
                
                if results:
                    processor.plot_results(results)
                    
                    # Removed: "Total speed detections: X" summary logging
                    logger.info("Ready for next command...")
                else:
                    logger.warning("No data captured or processed")
                    logger.info("Ready for next command...")
            else:
                logger.info("Unknown command. Available commands:")
                logger.info("  'Trig' - capture and process data")
                logger.info("  'quit' - exit")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        processor.reset_sensor()
        processor.disconnect()


if __name__ == "__main__":
    main()