import torch
import numpy as np
import time

class CoordinatorModule:
    def __init__(self, num_gpus=1, nvlink_threshold=0.7):
        """
        Initialize the coordinator module for multi-GPU coordination.
        
        Args:
            num_gpus: Number of GPUs to coordinate
            nvlink_threshold: GPU memory threshold to trigger coordination (0.0-1.0)
        """
        self.num_gpus = num_gpus
        self.nvlink_threshold = nvlink_threshold
        
        # Verify GPU availability
        self.available_gpus = min(torch.cuda.device_count(), num_gpus)
        if self.available_gpus < num_gpus:
            print(f"Warning: Requested {num_gpus} GPUs, but only {self.available_gpus} are available.")
        
        # Track GPU memory usage
        self.gpu_memory_usage = [0.0] * self.available_gpus
        
        # Track allocated blocks per GPU
        self.gpu_allocated_blocks = [set() for _ in range(self.available_gpus)]
        
        # Check NVLink connectivity
        self.nvlink_matrix = self._check_nvlink_connectivity()
    
    def _check_nvlink_connectivity(self):
        """
        Check NVLink connectivity between GPUs.
        
        Returns:
            Matrix of connectivity between GPUs
        """
        if self.available_gpus <= 1:
            return [[True]]
        
        # Try to use NVML to get connectivity information
        try:
            import pynvml
            pynvml.nvmlInit()
            
            matrix = [[False for _ in range(self.available_gpus)] for _ in range(self.available_gpus)]
            
            for i in range(self.available_gpus):
                # Every GPU is connected to itself
                matrix[i][i] = True
                
                handle_i = pynvml.nvmlDeviceGetHandleByIndex(i)
                for j in range(i+1, self.available_gpus):
                    handle_j = pynvml.nvmlDeviceGetHandleByIndex(j)
                    
                    try:
                        # Check P2P capability
                        link_type = pynvml.nvmlDeviceGetP2PStatus(handle_i, handle_j, pynvml.NVML_P2P_CAPABILITY_LINK_TYPE)
                        if link_type == pynvml.NVML_P2P_LINK_TYPE_NVLINK:
                            matrix[i][j] = True
                            matrix[j][i] = True
                    except:
                        pass
            
            pynvml.nvmlShutdown()
            return matrix
        except:
            # If NVML not available, assume all GPUs are connected
            matrix = [[i == j or True for i in range(self.available_gpus)] for j in range(self.available_gpus)]
            return matrix
    
    def get_gpu_memory_usage(self):
        """
        Get current GPU memory usage.
        
        Returns:
            List of GPU memory utilization (0.0-1.0)
        """
        usage = []
        for i in range(self.available_gpus):
            try:
                total_mem = torch.cuda.get_device_properties(i).total_memory
                reserved_mem = torch.cuda.memory_reserved(i)
                allocated_mem = torch.cuda.memory_allocated(i)
                
                # Compute utilization as fraction of total memory
                utilization = reserved_mem / total_mem
                usage.append(utilization)
            except:
                usage.append(0.0)
        
        self.gpu_memory_usage = usage
        return usage
    
    def balance_load(self, kv_blocks):
        """
        Balance load across multiple GPUs by moving blocks.
        
        Args:
            kv_blocks: Dictionary mapping block_id to (device, tensor)
            
        Returns:
            Dictionary of blocks to move: {block_id: target_gpu_id}
        """
        if self.available_gpus <= 1:
            return {}
        
        # Get current memory usage
        usage = self.get_gpu_memory_usage()
        
        # Check if any GPU is over threshold
        over_threshold = [i for i, u in enumerate(usage) if u > self.nvlink_threshold]
        under_threshold = [i for i, u in enumerate(usage) if u < self.nvlink_threshold]
        
        if not over_threshold or not under_threshold:
            return {}
        
        blocks_to_move = {}
        
        # For each GPU over threshold
        for from_gpu in over_threshold:
            # Find blocks on this GPU
            gpu_blocks = {
                bid: (dev, tensor) for bid, (dev, tensor) in kv_blocks.items()
                if dev == f"cuda:{from_gpu}"
            }
            
            if not gpu_blocks:
                continue
            
            # Find eligible target GPUs (under threshold and connected via NVLink)
            eligible_targets = [
                to_gpu for to_gpu in under_threshold
                if self.nvlink_matrix[from_gpu][to_gpu]
            ]
            
            if not eligible_targets:
                continue
            
            # Sort blocks by last access time (oldest first)
            sorted_blocks = sorted(
                gpu_blocks.keys(),
                key=lambda bid: kv_blocks.get(bid, (None, None, float('inf')))[2]
            )
            
            # Move blocks until under threshold or no more blocks
            blocks_to_move_count = int(len(sorted_blocks) * 0.3)  # Move up to 30% of blocks
            if blocks_to_move_count == 0 and sorted_blocks:
                blocks_to_move_count = 1  # Move at least one block
            
            for i in range(min(blocks_to_move_count, len(sorted_blocks))):
                # Round-robin distribution to target GPUs
                target_gpu = eligible_targets[i % len(eligible_targets)]
                blocks_to_move[sorted_blocks[i]] = target_gpu
        
        return blocks_to_move
    
    def select_gpu_for_block(self, block_size):
        """
        Select the optimal GPU for a new block.
        
        Args:
            block_size: Size of the block in bytes
            
        Returns:
            GPU device ID
        """
        # Get current memory usage
        usage = self.get_gpu_memory_usage()
        
        # Select GPU with lowest memory usage
        lowest_usage = float('inf')
        selected_gpu = 0
        
        for i, u in enumerate(usage):
            if u < lowest_usage:
                lowest_usage = u
                selected_gpu = i
        
        return selected_gpu
    
    def track_block(self, block_id, gpu_id):
        """Track a block allocation on a specific GPU."""
        if 0 <= gpu_id < self.available_gpus:
            self.gpu_allocated_blocks[gpu_id].add(block_id)
    
    def untrack_block(self, block_id, gpu_id):
        """Remove tracking for a block on a specific GPU."""
        if 0 <= gpu_id < self.available_gpus and block_id in self.gpu_allocated_blocks[gpu_id]:
            self.gpu_allocated_blocks[gpu_id].remove(block_id)
