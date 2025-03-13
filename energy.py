import threading
import time
import psutil
import numpy as np

class EnergyMonitor:
    def __init__(self, enabled=True, sampling_interval=0.1):
        """
        Initialize the energy monitor for tracking GPU power usage.
        
        Args:
            enabled: Whether energy monitoring is enabled
            sampling_interval: Interval for sampling power usage (seconds)
        """
        self.enabled = enabled
        self.sampling_interval = sampling_interval
        
        # Internal state
        self.running = False
        self.thread = None
        self.power_samples = []
        self.start_time = None
        self.end_time = None
        
        # Check if we have access to NVML for power monitoring
        self.nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_available = True
            self.pynvml = pynvml
            self.num_gpus = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]
        except (ImportError, Exception) as e:
            print(f"Warning: pynvml not available. Energy monitoring will be limited. Error: {e}")
            self.nvml_available = False
    
    def start_monitoring(self):
        """Start monitoring GPU power usage."""
        if not self.enabled:
            return
        
        if self.running:
            return
        
        self.running = True
        self.power_samples = []
        self.start_time = time.time()
        
        if self.nvml_available:
            # Start monitoring thread
            self.thread = threading.Thread(target=self._monitor_power)
            self.thread.daemon = True
            self.thread.start()
        else:
            # If NVML not available, use CPU usage as proxy
            self.thread = threading.Thread(target=self._monitor_cpu)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring GPU power usage."""
        if not self.running:
            return
        
        self.running = False
        self.end_time = time.time()
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def _monitor_power(self):
        """Monitor GPU power usage."""
        try:
            while self.running:
                sample_time = time.time()
                powers = []
                
                for handle in self.handles:
                    try:
                        power = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                        powers.append(power)
                    except Exception as e:
                        powers.append(0.0)
                
                self.power_samples.append((sample_time, powers))
                time.sleep(self.sampling_interval)
        except Exception as e:
            print(f"Error in power monitoring: {e}")
            self.running = False
    
    def _monitor_cpu(self):
        """Monitor CPU usage as proxy for energy usage."""
        try:
            while self.running:
                sample_time = time.time()
                cpu_percent = psutil.cpu_percent(interval=self.sampling_interval)
                self.power_samples.append((sample_time, [cpu_percent]))
        except Exception as e:
            print(f"Error in CPU monitoring: {e}")
            self.running = False
    
    def get_average_power(self):
        """Get average power usage during monitoring period."""
        if not self.power_samples:
            return [0.0]
        
        # Calculate average power per GPU
        powers = np.array([sample[1] for sample in self.power_samples])
        avg_powers = np.mean(powers, axis=0)
        
        return avg_powers.tolist()
    
    def get_total_energy(self):
        """Get total energy usage during monitoring period in joules."""
        if not self.power_samples or not self.start_time or not self.end_time:
            return [0.0]
        
        # Calculate average power
        avg_powers = self.get_average_power()
        
        # Calculate duration
        duration = self.end_time - self.start_time
        
        # Energy = Power × Time
        total_energy = [p * duration for p in avg_powers]
        
        return total_energy
    
    def get_efficiency_ratio(self, tokens_generated):
        """
        Calculate energy efficiency ratio (tokens/joule).
        
        Args:
            tokens_generated: Number of tokens generated during monitoring period
            
        Returns:
            List of efficiency ratios per GPU
        """
        total_energy = self.get_total_energy()
        
        # Avoid division by zero
        efficiency = [
            tokens_generated / max(e, 0.001) for e in total_energy
        ]
        
        return efficiency
    
    def get_power_state(self):
        """
        Get current power state (high/medium/low) based on recent usage.
        
        Returns:
            Power state: "high", "medium", or "low"
        """
        if not self.power_samples:
            return "medium"
        
        # Get recent samples (last 10)
        recent_samples = self.power_samples[-10:]
        
        if not recent_samples:
            return "medium"
        
        # Calculate average recent power
        recent_powers = np.array([sample[1] for sample in recent_samples])
        avg_recent_power = np.mean(recent_powers)
        
        # Determine state based on percentage of max power
        if self.nvml_available:
            try:
                # Get max power limit
                max_powers = []
                for handle in self.handles:
                    try:
                        power_limit = self.pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # mW to W
                        max_powers.append(power_limit)
                    except:
                        max_powers.append(300.0)  # Default assumption
                
                avg_max_power = np.mean(max_powers)
                power_ratio = avg_recent_power / avg_max_power
                
                if power_ratio > 0.7:
                    return "high"
                elif power_ratio > 0.3:
                    return "medium"
                else:
                    return "low"
            except:
                pass
        
        # Fallback to simple heuristic
        if len(self.power_samples) > 100:
            all_powers = np.array([sample[1] for sample in self.power_samples])
            all_avg = np.mean(all_powers)
            
            if avg_recent_power > all_avg * 1.2:
                return "high"
            elif avg_recent_power < all_avg * 0.8:
                return "low"
        
        return "medium"
