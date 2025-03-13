import os
import torch
import time
import threading

class NVMEOffloadedTensor:
    """Placeholder class for tensors offloaded to NVMe."""
    def __init__(self, file_path, shape, dtype):
        self.file_path = file_path
        self.shape = shape
        self.dtype = dtype
        self._tensor = None
    
    def __repr__(self):
        return f"NVMEOffloadedTensor(shape={self.shape}, dtype={self.dtype}, path={self.file_path})"

class OffloaderModule:
    def __init__(self, target="cpu", nvme_path=None):
        """
        Initialize the offloader module for KV cache offloading.
        
        Args:
            target: Target for offloading ("cpu" or "nvme")
            nvme_path: Path to NVMe storage location (if target is "nvme")
        """
        self.target = target
        self.nvme_path = nvme_path or os.path.join(os.getcwd(), "nvme_offload")
        
        # Create directory for NVMe offloading if needed
        if target == "nvme" and not os.path.exists(self.nvme_path):
            os.makedirs(self.nvme_path)
        
        # Track offloaded blocks for cleanup
        self.offloaded_blocks = {}  # Maps block_id to file_path
        
        # Prefetch queue
        self.prefetch_queue = []
        self.prefetch_thread = None
        self.prefetch_lock = threading.Lock()
        self.prefetch_running = False
    
    def offload(self, tensor, block_id=None, target=None):
        """
        Offload a tensor to CPU or NVMe.
        
        Args:
            tensor: Tensor to offload
            block_id: Optional block ID for tracking
            target: Override the default target
            
        Returns:
            Tuple of (new_device, new_tensor, block_id)
        """
        target = target or self.target
        if block_id is None:
            block_id = id(tensor)
        
        if target == "cpu":
            # Offload to CPU memory
            new_tensor = tensor.cpu()
            new_device = "cpu"
        
        elif target == "nvme":
            # Offload to NVMe storage
            file_path = os.path.join(self.nvme_path, f"block_{block_id}.pt")
            
            # Save to disk
            cpu_tensor = tensor.cpu()
            torch.save(cpu_tensor, file_path)
            
            # Free GPU memory
            del tensor
            torch.cuda.empty_cache()
            
            # Create a reference object
            new_tensor = NVMEOffloadedTensor(file_path, cpu_tensor.shape, cpu_tensor.dtype)
            new_device = "nvme"
            
            # Track for cleanup
            self.offloaded_blocks[block_id] = file_path
        
        else:
            raise ValueError(f"Unknown offload target: {target}")
        
        return new_device, new_tensor, block_id
    
    def fetch(self, tensor_or_path, device_id=0, async_mode=False):
        """
        Fetch a tensor from CPU or NVMe back to GPU.
        
        Args:
            tensor_or_path: Tensor or NVMEOffloadedTensor to fetch
            device_id: Target GPU device ID
            async_mode: Whether to fetch asynchronously
            
        Returns:
            The fetched tensor on GPU
        """
        if isinstance(tensor_or_path, NVMEOffloadedTensor):
            # Load from NVMe
            if async_mode:
                # Start async load and return a dummy tensor as a placeholder
                self._async_fetch(tensor_or_path, device_id)
                # Return a placeholder that will be replaced later
                dummy = torch.zeros(1, device=f"cuda:{device_id}")
                return dummy
            else:
                # Sync load from disk
                tensor = torch.load(tensor_or_path.file_path)
                return tensor.to(f"cuda:{device_id}")
        elif isinstance(tensor_or_path, torch.Tensor):
            # Move tensor to GPU
            return tensor_or_path.to(f"cuda:{device_id}")
        else:
            raise ValueError(f"Unknown tensor type: {type(tensor_or_path)}")
    
    def _async_fetch(self, nvme_tensor, device_id):
        """Schedule asynchronous fetch from NVMe."""
        with self.prefetch_lock:
            self.prefetch_queue.append((nvme_tensor, device_id))
            
            # Start prefetch thread if not running
            if not self.prefetch_running:
                self.prefetch_running = True
                self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
                self.prefetch_thread.daemon = True
                self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Worker thread for handling prefetch requests."""
        try:
            while True:
                # Get next item from queue
                with self.prefetch_lock:
                    if not self.prefetch_queue:
                        self.prefetch_running = False
                        break
                    nvme_tensor, device_id = self.prefetch_queue.pop(0)
                
                # Load from disk
                tensor = torch.load(nvme_tensor.file_path)
                
                # Move to GPU
                tensor = tensor.to(f"cuda:{device_id}")
                
                # Store result for later retrieval
                nvme_tensor._tensor = tensor
        except Exception as e:
            print(f"Error in prefetch worker: {e}")
            self.prefetch_running = False
    
    def schedule_prefetch(self, block_id, device_id=0):
        """
        Schedule a block to be prefetched from offloaded storage to GPU.
        
        Args:
            block_id: ID of the block to prefetch
            device_id: Target GPU device ID
        """
        if block_id in self.offloaded_blocks:
            file_path = self.offloaded_blocks[block_id]
            shape = None
            dtype = None
            
            # Try to get shape and dtype from file metadata
            try:
                tensor_meta = torch.load(file_path, map_location="cpu")
                shape = tensor_meta.shape
                dtype = tensor_meta.dtype
                del tensor_meta
            except:
                pass
            
            # Create a placeholder and schedule prefetch
            nvme_tensor = NVMEOffloadedTensor(file_path, shape, dtype)
            self._async_fetch(nvme_tensor, device_id)
    
    def cleanup(self):
        """Clean up offloaded blocks."""
        for file_path in self.offloaded_blocks.values():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        self.offloaded_blocks = {}
