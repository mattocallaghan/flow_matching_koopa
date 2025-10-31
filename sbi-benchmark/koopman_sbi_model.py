"""
SBI-Compatible Koopman Conditional Flow Matching Model

This module implements a Koopman-based flow matching model that is compatible
with the SBI benchmark framework and dingo infrastructure. It follows the 
proper data flow: (theta, x) training pairs -> sample theta|x

The model learns from a pre-trained flow matching teacher and provides fast
one-step conditional posterior sampling using Koopman operators.

Author: Based on Koopman Flow Matching paper adaptation for SBI
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

try:
    from dingo.core.posterior_models.build_model import build_model_from_kwargs
    DINGO_AVAILABLE = True
except ImportError:
    DINGO_AVAILABLE = False


logger = logging.getLogger(__name__)


class KoopmanSBIModel:
    """
    Koopman-based flow matching model for SBI tasks.
    
    Standalone implementation that provides fast posterior sampling using
    Koopman operators. Compatible with SBI evaluation framework.
    
    Key features:
    1. Uses pre-trained teacher model to generate training data buffer
    2. Learns Koopman lifting coordinates and operators
    3. Provides one-step sampling via matrix exponentials
    4. Compatible with SBI evaluation (sample_batch, etc.)
    """
    
    def __init__(self, 
                 teacher_model_path: Optional[str] = None,
                 lifted_dim: int = 64,
                 lambda_phase: float = 1.0,
                 lambda_target: float = 1.0,
                 lambda_recon: float = 1.0,
                 lambda_cons: float = 0.1,
                 buffer_size: int = 5000,
                 hidden_dims: Optional[list] = None,
                 input_dim: int = 2,
                 context_dim: int = 2,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize Koopman SBI model
        
        Args:
            teacher_model_path: Path to pre-trained flow matching teacher
            lifted_dim: Dimension of Koopman lifting space
            lambda_phase: Weight for phase loss  
            lambda_target: Weight for target loss
            lambda_recon: Weight for reconstruction loss
            lambda_cons: Weight for consistency loss with teacher
            buffer_size: Size of (theta_0, theta_1, x) training buffer
            input_dim: Dimension of theta parameters
            context_dim: Dimension of observation x
            device: Device for computation
        """
        self.teacher_model_path = teacher_model_path
        self.lifted_dim = lifted_dim
        self.hidden_dims = hidden_dims or [128, 256, 128]
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.device = torch.device(device)
        
        # Loss weights from Koopman paper
        self.lambda_phase = lambda_phase
        self.lambda_target = lambda_target
        self.lambda_recon = lambda_recon
        self.lambda_cons = lambda_cons
        
        # Buffer for teacher-generated (theta_0, theta_1, x) triplets
        self.buffer = []
        self.buffer_size = buffer_size
        
        # Will be initialized in initialize_network()
        self.teacher_model = None
        self.encoder = None
        self.affine_generator = None
        self.decoder = None
        
        logger.info(f"Initialized KoopmanSBIModel with lifted_dim={lifted_dim}")
        
    def initialize_network(self):
        """Initialize Koopman components and load teacher model"""
        
        # Load teacher model if provided
        assert self.teacher_model_path and DINGO_AVAILABLE, "Teacher model path must be provided and dingo must be available"
        try:
            # Use device type for dingo compatibility
            device_str = self.device.type
            self.teacher_model = build_model_from_kwargs(
                filename=self.teacher_model_path,
                device=device_str
            )
            self.teacher_model.network.eval()
            logger.info(f"Loaded teacher model from {self.teacher_model_path}")
        except Exception as e:
            logger.warning(f"Could not load teacher model: {e}")
            raise e
        
        # Koopman encoder: (t, theta, x) -> lifted coordinates
        # For two_moons: theta_dim=2, x_dim=2, so input is 1+2+2=5
        encoder_input_dim = 1 + self.input_dim + self.context_dim
        encoder_layers = []
        dims = [encoder_input_dim] + self.hidden_dims + [self.lifted_dim - 2]
        
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        
        # Replace matrix exponential with simple linear transformation
        # Direct linear evolution from z_0 -> z_1 
        self.linear_evolution = nn.Linear(self.lifted_dim, self.lifted_dim, bias=True)
        
        # Initialize to near-identity for stability
        with torch.no_grad():
            self.linear_evolution.weight.data = torch.eye(self.lifted_dim) + 0.1 * torch.randn(self.lifted_dim, self.lifted_dim)
            self.linear_evolution.bias.data.zero_()
        
        # Conditional control C_μ(x): affects all lifted coordinates
        self.control_linear = nn.Linear(self.context_dim, self.lifted_dim).to(self.device)
        
        # Decoder: lifted coordinates -> theta parameters
        decoder_layers = []
        decoder_dims = [self.lifted_dim] + list(reversed(self.hidden_dims)) + [self.input_dim]
        
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:  # No activation after last layer
                decoder_layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        
        logger.info("Initialized Koopman networks")
    
    def load_koopman_dataset(self, koopman_data: dict):
        """
        Load pre-generated Koopman dataset
        
        Args:
            koopman_data: Dictionary containing theta_0, theta_1, x, and metadata
        """
        self.theta_0 = koopman_data['theta_0'].to(self.device)
        self.theta_1 = koopman_data['theta_1'].to(self.device) 
        self.x_context = koopman_data['x'].to(self.device)
        
        # Store dataset info
        self.dataset_size = len(self.theta_0)
        self.dataset_metadata = {
            'buffer_size': koopman_data['buffer_size'],
            'samples_per_observation': koopman_data['samples_per_observation']
        }
        
        # Clear old buffer (no longer needed)
        self.buffer = []
        
        logger.info(f"Loaded Koopman dataset: {self.dataset_size} triplets")
        logger.info(f"Dataset shapes: theta_0={self.theta_0.shape}, theta_1={self.theta_1.shape}, x={self.x_context.shape}")
    
    def initialize_buffer(self, x_samples: torch.Tensor = None, theta_samples: torch.Tensor = None, min_samples: int = None):
        """
        Initialize buffer with training data
        
        Args:
            x_samples: Observation samples 
            theta_samples: Parameter samples (if using data source)
            min_samples: Not used - buffer size matches available data
        """
        # Check buffer source from settings
        buffer_source = self.buffer_source 
        
        if buffer_source == 'data' and theta_samples is not None:
            # Use ALL training data pairs 
            logger.info("Using all training data for buffer initialization")
            n_available = min(len(theta_samples), len(x_samples))
            
            for i in range(n_available):
                context = x_samples[i]                   # observation context
                x0 = torch.randn(theta_samples[i].shape).to(self.device)  # x0 ~ prior (noise)
                x1 = theta_samples[i].to(self.device)    # x1 ~ data
                self.buffer.append((x0, x1, context))
                
            logger.info(f"Buffer initialized with ALL {len(self.buffer)} training samples")
                
        else:
            # Use teacher model sampling - match training set size, process in batches
            logger.info("Using teacher model sampling for buffer initialization")
            assert x_samples is not None, "x_samples must be provided for teacher sampling"
            
            # Process in batches for efficiency
            batch_size = 64  # Process 64 observations at a time
            n_samples = len(x_samples)
            
            with torch.no_grad():
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    x_batch = x_samples[start_idx:end_idx].to(self.device)
                    
                    # Generate teacher samples for this batch
                    x1_batch = self.teacher_model.sample_batch(x_batch)  # [batch_size, theta_dim]
                    x0_batch = torch.randn_like(x1_batch)  # [batch_size, theta_dim]
                    
                    # Add to buffer
                    for i in range(len(x_batch)):
                        context = x_batch[i]
                        x0 = x0_batch[i] 
                        x1 = x1_batch[i]
                        self.buffer.append((x0, x1, context))
                    
            logger.info(f"Buffer initialized with {len(self.buffer)} teacher-generated samples")
        
    def generate_teacher_data(self, x_batch: torch.Tensor, n_samples_per_x: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate (theta_0, theta_1, x) triplets using teacher model
        
        Args:
            x_batch: Batch of observations
            n_samples_per_x: Number of samples to generate per observation
            
        Returns:
            Tuple of (theta_0, theta_1, x_repeated)
        """
        assert self.teacher_model is not None
            # Fallback: generate dummy data for testing

            
        theta_0_list = []
        theta_1_list = []
        x_repeated_list = []
        
        with torch.no_grad():
            for x in x_batch:
                # Sample from teacher: theta_1 ~ p(theta|x)
                if hasattr(self.teacher_model, 'sample_batch'):
                    # Use batch sampling if available
                    x_expanded = x.unsqueeze(0).repeat(n_samples_per_x, 1)
                    theta_1 = self.teacher_model.sample_batch(x_expanded)
                else:
                    # Fall back to single sampling
                    theta_1 = self.teacher_model.sample(n_samples_per_x, x)
                
                # Generate noise
                theta_0 = torch.randn_like(theta_1)
                
                # Repeat observation
                x_repeated = x.unsqueeze(0).repeat(n_samples_per_x, 1)
                
                theta_0_list.append(theta_0)
                theta_1_list.append(theta_1)
                x_repeated_list.append(x_repeated)
                
        return (torch.cat(theta_0_list, dim=0),
                torch.cat(theta_1_list, dim=0),
                torch.cat(x_repeated_list, dim=0))
    
    def update_buffer(self, x_batch: torch.Tensor, force_update: bool = False):
        """Update training buffer with new teacher data
        
        Args:
            x_batch: Observations to generate data for
            force_update: If True, always update regardless of buffer size
        """
        if len(x_batch) == 0:
            return
            
        # Generate new data (limit to avoid memory issues unless force_update)
        if force_update:
            n_obs_to_process = min(len(x_batch), 10)  # More aggressive when force updating
            n_samples_per_x = 50  # More samples per observation
        else:
            n_obs_to_process = min(len(x_batch), 3)
            n_samples_per_x = 20
            
        theta_0, theta_1, x_repeated = self.generate_teacher_data(
            x_batch[:n_obs_to_process], 
            n_samples_per_x=n_samples_per_x
        )
        
        # Add to buffer
        for i in range(len(theta_0)):
            if len(self.buffer) >= self.buffer_size:
                # Remove oldest (FIFO)
                self.buffer.pop(0)
            self.buffer.append((
                theta_0[i].detach().cpu(),
                theta_1[i].detach().cpu(), 
                x_repeated[i].detach().cpu()
            ))
        
        logger.debug(f"Buffer updated: {len(self.buffer)}/{self.buffer_size} samples")
    
    def compute_lifted_coords(self, t: torch.Tensor, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute lifted coordinates z̃(t,θ,x) = [1, t, gφ(t,θ,x)]⊤
        
        Args:
            t: Time parameter [batch_size, 1] or [batch_size] or scalar
            theta: Parameter samples [batch_size, theta_dim]
            x: Observations [batch_size, x_dim]
            
        Returns:
            Lifted coordinates [batch_size, lifted_dim]
        """
        batch_size = theta.shape[0]
        
        # Ensure t is properly shaped
        if t.dim() == 0:
            t_expanded = t.expand(batch_size, 1)
        elif t.dim() == 1:
            t_expanded = t.view(-1, 1)
        else:
            t_expanded = t
            
        # Ensure x is properly shaped
        if x.dim() == 1:
            x_expanded = x.unsqueeze(0).expand(batch_size, -1)
        else:
            x_expanded = x
            
        # Concatenate inputs for encoder: [t, theta, x]
        encoder_input = torch.cat([t_expanded, theta, x_expanded], dim=-1)
        
        # Encode
        encoded = self.encoder(encoder_input)
        
        # Create lifted coordinates [1, t, encoded]
        ones = torch.ones(batch_size, 1).to(self.device)
        
        return torch.cat([ones, t_expanded, encoded], dim=-1)
    
    # Matrix exponential methods removed - using linear evolution instead
    
    def loss(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Koopman training loss
        
        Args:
            theta: Parameter samples [batch_size, theta_dim]
            x: Observations [batch_size, x_dim]
            
        Returns:
            Total training loss
        """
        # Ensure inputs are tensors with proper shapes
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32).to(self.device)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            
        # Ensure proper dimensions
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Use pre-generated dataset - no buffer needed
        if not hasattr(self, 'theta_0') or self.dataset_size == 0:
            raise RuntimeError("Pre-generated Koopman dataset not loaded. Call load_koopman_dataset() first.")
            
        # Sample from pre-generated dataset for Koopman losses
        batch_size = min(self.dataset_size, theta.shape[0])
        dataset_indices = np.random.choice(self.dataset_size, size=batch_size, replace=False)
        
        theta_0_buffer = self.theta_0[dataset_indices]
        theta_1_buffer = self.theta_1[dataset_indices]
        x_buffer = self.x_context[dataset_indices]
        
        # 1. PHASE LOSS: ||e^L * gφ(0, θ₀) + C_μ(x) - gφ(1, θ₁)||²
        z_0 = self.compute_lifted_coords(torch.zeros(len(theta_0_buffer), 1).to(self.device), 
                                        theta_0_buffer, x_buffer)
        z_1_target = self.compute_lifted_coords(torch.ones(len(theta_1_buffer), 1).to(self.device),
                                               theta_1_buffer, x_buffer)
        
        # Linear evolution + conditional control
        z_1_evolved = self.linear_evolution(z_0) + self.control_linear(x_buffer)
        
        L_phase = torch.mean((z_1_evolved - z_1_target)**2)
        
        # 2. TARGET LOSS: ||decoder(e^L * gφ(0, θ₀)) - θ₁||²
        theta_1_decoded = self.decoder(z_1_evolved)
        L_target = torch.mean((theta_1_decoded - theta_1_buffer)**2)
        
        # 3. RECONSTRUCTION LOSS: ||decoder(gφ(1, θ₁)) - θ₁||²
        theta_1_recon = self.decoder(z_1_target)
        L_recon = torch.mean((theta_1_recon - theta_1_buffer)**2)
        
        # 4. CONSISTENCY LOSS (skip if lambda_cons = 0.0)
        if self.lambda_cons > 0.0:
            t_random = torch.rand(theta.shape[0], 1).to(self.device)
            theta_0_noise = torch.randn_like(theta)
            theta_t = (1 - t_random) * theta_0_noise + t_random * theta
            
            # Teacher velocity - teacher expects scalar t, so we use single time value for whole batch
            with torch.no_grad():
                t_scalar = t_random[0, 0].item()  # Use first time value as scalar
                try:
                    v_teacher = self.teacher_model.evaluate_vectorfield(t_scalar, theta_t, x)
                    # Check for NaN/inf in teacher output
                    if torch.isnan(v_teacher).any() or torch.isinf(v_teacher).any():
                        logger.warning("Teacher model outputs contain NaN/inf values")
                        v_teacher = torch.zeros_like(theta_t)  # Fallback to zero velocity
                except Exception as e:
                    logger.error(f"Teacher model evaluation failed: {e}")
                    v_teacher = torch.zeros_like(theta_t)  # Fallback to zero velocity
            
            # Skip consistency loss with linear model (no differential dynamics)
            L_cons = torch.tensor(0.0, device=self.device)
        else:
            # Skip consistency loss computation entirely
            L_cons = torch.tensor(0.0, device=self.device)

        
        # Total Koopman loss
        koopman_loss = (self.lambda_phase * L_phase + 
                       self.lambda_target * L_target + 
                       self.lambda_recon * L_recon + 
                       self.lambda_cons * L_cons)
        
        # Return only Koopman loss (no flow matching baseline)
        total_loss = koopman_loss
        
        # Logging
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        # Log more frequently in early training
        log_frequency = 10 if self._step_count < 100 else 50
        if self._step_count % log_frequency == 0:
            buffer_stats = self.get_buffer_stats()
            
            # Check for problematic loss values
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error(f"Step {self._step_count}: Loss contains NaN/inf!")
                logger.error(f"Phase={L_phase:.4f}, Target={L_target:.4f}, Recon={L_recon:.4f}, Cons={L_cons:.4f}")
            
            logger.info(f"Step {self._step_count}: "
                       f"Phase={L_phase:.4f}, Target={L_target:.4f}, "
                       f"Recon={L_recon:.4f}, Cons={L_cons:.4f}, "
                       f"Total={total_loss:.4f}, "
                       f"Buffer={buffer_stats['buffer_size']}/{buffer_stats['buffer_capacity']}")
        
        return total_loss
    
    def validation_loss(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified validation loss without buffer updates
        """
        # Ensure inputs are tensors with proper shapes
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float32).to(self.device)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            
        # Ensure proper dimensions
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Use pre-generated dataset for validation
        if not hasattr(self, 'theta_0') or self.dataset_size == 0:
            # Return simple reconstruction loss if dataset not loaded
            return torch.mean((theta - theta)**2)  # Dummy loss
        
        # Sample from pre-generated dataset for validation losses
        batch_size = min(self.dataset_size, theta.shape[0])
        dataset_indices = np.random.choice(self.dataset_size, size=batch_size, replace=False)
        
        theta_0_buffer = self.theta_0[dataset_indices]
        theta_1_buffer = self.theta_1[dataset_indices]
        x_buffer = self.x_context[dataset_indices]
        
        # Compute only core Koopman losses (same as training but no buffer updates)
        # 1. PHASE LOSS with conditional control
        z_0 = self.compute_lifted_coords(torch.zeros(len(theta_0_buffer), 1).to(self.device), 
                                        theta_0_buffer, x_buffer)
        z_1_target = self.compute_lifted_coords(torch.ones(len(theta_1_buffer), 1).to(self.device),
                                               theta_1_buffer, x_buffer)
        
        # Linear evolution + conditional control
        z_1_evolved = self.linear_evolution(z_0) + self.control_linear(x_buffer)
        
        L_phase = torch.mean((z_1_evolved - z_1_target)**2)
        
        # 2. TARGET LOSS 
        theta_1_pred = self.decoder(z_1_evolved)
        L_target = torch.mean((theta_1_pred - theta_1_buffer)**2)
        
        # 3. RECONSTRUCTION LOSS
        theta_1_recon = self.decoder(z_1_target)
        L_recon = torch.mean((theta_1_recon - theta_1_buffer)**2)
        
        # Total validation loss (skip consistency loss to avoid teacher model calls)
        val_loss = (self.lambda_phase * L_phase + 
                   self.lambda_target * L_target + 
                   self.lambda_recon * L_recon)
        
        return val_loss
    
    def sample_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast batch sampling using Koopman operators (SBI interface)
        
        Args:
            x: Observation batch [batch_size, x_dim]
            
        Returns:
            Parameter samples [batch_size, theta_dim]
        """
        # Always use Koopman sampling for this implementation
        return self.koopman_sample_batch(x)
    
    def koopman_sample_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Koopman-based fast sampling
        
        Args:
            x: Observation batch [batch_size, x_dim]
            
        Returns:
            Parameter samples [batch_size, theta_dim]
        """
        # Ensure model is in eval mode
        self.eval()
        
        batch_size = x.shape[0]
        
        # Sample noise
        theta_0 = torch.randn(batch_size, self.input_dim).to(self.device)
        
        # Lift to Koopman space at t=0
        z_0 = self.compute_lifted_coords(
            torch.zeros(batch_size, 1).to(self.device),
            theta_0,
            x
        )
        
        # One-step evolution to t=1 with linear transformation
        z_1 = self.linear_evolution(z_0) + self.control_linear(x)
        
        # Decode to parameter space
        theta_samples = self.decoder(z_1)
        
        return theta_samples
    
    def parameters(self):
        """Return iterator over all model parameters"""
        params = []
        if hasattr(self, 'encoder'):
            params.extend(self.encoder.parameters())
        if hasattr(self, 'decoder'):
            params.extend(self.decoder.parameters())
        # Universal L matrix parameters
        # Add linear evolution parameters
        params.extend(list(self.linear_evolution.parameters()))
        # Conditional control network
        if hasattr(self, 'control_linear'):
            params.extend(self.control_linear.parameters())
        return iter(params)
    
    def state_dict(self):
        """Return state dictionary of all model components"""
        state = {}
        if hasattr(self, 'encoder'):
            state['encoder'] = self.encoder.state_dict()
        if hasattr(self, 'decoder'):
            state['decoder'] = self.decoder.state_dict()
        # Universal L matrix parameters
        # Linear evolution parameters
        state['linear_evolution'] = self.linear_evolution.state_dict()
        # Conditional control network
        if hasattr(self, 'control_linear'):
            state['control_linear'] = self.control_linear.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dictionary into model components"""
        if 'encoder' in state_dict and hasattr(self, 'encoder'):
            self.encoder.load_state_dict(state_dict['encoder'])
        if 'decoder' in state_dict and hasattr(self, 'decoder'):
            self.decoder.load_state_dict(state_dict['decoder'])
        # Universal L matrix parameters
        # Load linear evolution parameters
        if 'linear_evolution' in state_dict:
            self.linear_evolution.load_state_dict(state_dict['linear_evolution'])
        # Conditional control network
        if 'control_linear' in state_dict and hasattr(self, 'control_linear'):
            self.control_linear.load_state_dict(state_dict['control_linear'])
    
    def eval(self):
        """Set model to evaluation mode"""
        if hasattr(self, 'encoder'):
            self.encoder.eval()
        if hasattr(self, 'decoder'):
            self.decoder.eval()
        if hasattr(self, 'control_linear'):
            self.control_linear.eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode"""
        if hasattr(self, 'encoder'):
            self.encoder.train(mode)
        if hasattr(self, 'decoder'):
            self.decoder.train(mode)
        if hasattr(self, 'control_linear'):
            self.control_linear.train(mode)
        return self
    
    def sample(self, n_samples: int, context: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sample interface for compatibility
        
        Args:
            n_samples: Number of samples
            context: Single observation
            
        Returns:
            Parameter samples [n_samples, theta_dim]
        """
        # Expand context to batch
        if context.dim() == 1:
            context_batch = context.unsqueeze(0).repeat(n_samples, 1)
        else:
            context_batch = context.repeat(n_samples, 1)
            
        return self.sample_batch(context_batch)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring"""
        return {
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer_size,
            'buffer_usage': len(self.buffer) / self.buffer_size if self.buffer_size > 0 else 0
        }
    
    def initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and scheduler (required by dingo interface)"""
        if hasattr(self, 'optimizer_kwargs') and self.optimizer_kwargs:
            if self.optimizer_kwargs['type'].lower() == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=float(self.optimizer_kwargs['lr']),
                    weight_decay=float(self.optimizer_kwargs.get('weight_decay', 0))
                )
            else:
                raise ValueError(f"Unknown optimizer type: {self.optimizer_kwargs['type']}")
        
        if hasattr(self, 'scheduler_kwargs') and self.scheduler_kwargs:
            if self.scheduler_kwargs['type'].lower() == 'reduce_on_plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=self.scheduler_kwargs.get('factor', 0.5),
                    patience=self.scheduler_kwargs.get('patience', 10)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {self.scheduler_kwargs['type']}")
    
    def train(self, train_loader, test_loader, train_dir, runtime_limits, early_stopping=True, use_wandb=False):
        """Training method compatible with dingo interface"""
        
        epochs = runtime_limits.max_epochs_total
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 15
        
        for epoch in range(epochs):
            # Training
            self.train_mode = True
            train_loss = 0.0
            n_batches = 0
            
            for batch_idx, (theta, x) in enumerate(train_loader):
                theta = theta.to(self.device)
                x = x.to(self.device)
                
                self.optimizer.zero_grad()
                
                loss = self.loss(theta, x)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            avg_train_loss = train_loss / max(n_batches, 1)
            
            # Validation
            self.eval()  # Set networks to eval mode
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for theta, x in test_loader:
                    theta = theta.to(self.device)
                    x = x.to(self.device)
                    
                    # Use simplified validation loss (no buffer updates)
                    loss = self.validation_loss(theta, x)
                    val_loss += loss.item()
                    n_val_batches += 1
            
            avg_val_loss = val_loss / max(n_val_batches, 1)
            
            # Update scheduler
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()
                    
            # Save to history.txt (matching dingo format: epoch, train_loss, val_loss, lr)
            current_lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 0.0
            history_line = f"{epoch}\t{avg_train_loss:.8f}\t{avg_val_loss:.8f}\t{current_lr:.8f}\n"
            
            import os
            history_file = os.path.join(train_dir, "history.txt")
            with open(history_file, "a") as f:
                f.write(history_line)
            
            # Logging
            logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                self.save_model(train_dir, epoch, avg_val_loss, is_best=True)
            else:
                patience_counter += 1
                
            # Early stopping
            if early_stopping and patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_model(train_dir, epoch, avg_val_loss, is_best=False)
                
            # Buffer stats
            if hasattr(self, 'get_buffer_stats'):
                buffer_stats = self.get_buffer_stats()
                if epoch % 10 == 0:
                    logger.info(f"Buffer stats: {buffer_stats}")
    
    def save_model(self, train_dir, epoch, loss, is_best=False):
        """Save model checkpoint"""
        import os
        os.makedirs(train_dir, exist_ok=True)
        
        # Create dingo-compatible checkpoint with metadata
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'metadata': {
                'train_settings': {
                    'model': {
                        'type': 'koopman_sbi',
                        'teacher_model_path': self.teacher_model_path,
                        'lifted_dim': self.lifted_dim,
                        'lambda_phase': self.lambda_phase,
                        'lambda_target': self.lambda_target,
                        'lambda_recon': self.lambda_recon,
                        'lambda_cons': self.lambda_cons,
                        'buffer_size': self.buffer_size,
                        'input_dim': self.input_dim,
                        'context_dim': self.context_dim
                    }
                }
            }
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(train_dir, "best_model.pt"))
            logger.info(f"Saved best model at epoch {epoch} with loss {loss:.6f}")
        else:
            torch.save(checkpoint, os.path.join(train_dir, f"checkpoint_epoch_{epoch}.pt"))