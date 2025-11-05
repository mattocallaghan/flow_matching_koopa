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
                 use_matrix_exponential: bool = False,
                 buffer_size: int = 5000,
                 hidden_dims: Optional[list] = None,
                 input_dim: int = 2,
                 context_dim: int = 2,
                 device: str = 'cpu',
                 teacher_mode: bool = True,
                 sigma_min: float = 0.0001,
                 **kwargs):
        """
        Initialize Koopman SBI model
        
        Args:
            teacher_model_path: Path to pre-trained flow matching teacher (for teacher_mode=True)
            lifted_dim: Dimension of Koopman lifting space
            lambda_phase: Weight for phase loss  
            lambda_target: Weight for target loss
            lambda_recon: Weight for reconstruction loss
            lambda_cons: Weight for consistency loss with teacher
            use_matrix_exponential: If True, use matrix exponential approach; if False, use linear layer
            buffer_size: Size of (theta_0, theta_1, x) training buffer
            input_dim: Dimension of theta parameters
            context_dim: Dimension of observation x
            device: Device for computation
            teacher_mode: If True, use pre-trained teacher model; if False, use direct flow matching
            sigma_min: Minimum noise level for flow matching scheduler
        """
        self.teacher_model_path = teacher_model_path
        self.lifted_dim = lifted_dim
        self.hidden_dims = hidden_dims or [128, 256, 128]
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.device = torch.device(device)
        
        # Mode selection
        self.teacher_mode = teacher_mode
        
        # Evolution method selection
        self.use_matrix_exponential = use_matrix_exponential
        
        # Flow matching parameters
        self.sigma_min = sigma_min
        
        # Loss weights from Koopman paper
        self.lambda_phase = lambda_phase
        self.lambda_target = lambda_target
        self.lambda_recon = lambda_recon
        self.lambda_cons = lambda_cons
        
        # Buffer for generated (theta_0, theta_1, x) triplets
        self.buffer = []
        self.buffer_size = buffer_size
        
        # Will be initialized in initialize_network()
        self.teacher_model = None
        self.encoder = None
        self.affine_generator = None
        self.decoder = None
        
        logger.info(f"Initialized KoopmanSBIModel with lifted_dim={lifted_dim}, teacher_mode={teacher_mode}")
        
    def initialize_network(self):
        """Initialize Koopman components and optionally load teacher model"""
        
        # Load teacher model if in teacher mode
        if self.teacher_mode:
            assert self.teacher_model_path and DINGO_AVAILABLE, "Teacher model path must be provided and dingo must be available in teacher mode"
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
        else:
            logger.info("Using direct flow matching mode (no teacher model required)")
        
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
        
        if self.use_matrix_exponential:
            # Matrix exponential approach with block-diagonal structure
            # The generator affects only coordinates [2:] (excluding [1,t] components)
            self.internal_dim = self.lifted_dim - 2
            
            # Ensure internal dimension is even for block-diagonal structure
            if self.internal_dim % 2 != 0:
                raise ValueError(f"Internal dimension {self.internal_dim} must be even for block-diagonal structure")
            
            self.n_blocks = self.internal_dim // 2
            
            # Conditional block-diagonal generator: parameters depend on observation x
            # Networks to generate α and β parameters from observation x
            self.alpha_net = nn.Sequential(
                nn.Linear(self.context_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_blocks)
            ).to(self.device)
            
            self.beta_net = nn.Sequential(
                nn.Linear(self.context_dim, 64), 
                nn.ReLU(),
                nn.Linear(64, self.n_blocks)
            ).to(self.device)
            
            # A_gt: Coupling between internal coordinates and time (also conditional)
            self.Agt_net = nn.Sequential(
                nn.Linear(self.context_dim, 64),
                nn.ReLU(), 
                nn.Linear(64, self.internal_dim)
            ).to(self.device)
            
            # b_g: Bias vector for internal coordinates (also conditional)
            self.bg_net = nn.Sequential(
                nn.Linear(self.context_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.internal_dim)
            ).to(self.device)
                
        else:
            # Linear evolution approach - direct transformation
            self.linear_evolution = nn.Linear(self.lifted_dim, self.lifted_dim, bias=True)
            
            # Initialize to near-identity for stability
            with torch.no_grad():
                self.linear_evolution.weight.data = torch.eye(self.lifted_dim) + 0.1 * torch.randn(self.lifted_dim, self.lifted_dim)
                self.linear_evolution.bias.data.zero_()
        
        # Note: Removed additive control - conditioning now happens through matrix parameters
        
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
    
    def flow_matching_scheduler(self, t: torch.Tensor) -> torch.Tensor:
        """
        Flow matching noise scheduler: σ(t) = σ_min
        
        Args:
            t: Time values [batch_size] or scalar
            
        Returns:
            Noise levels [batch_size] or scalar
        """
        if isinstance(t, (int, float)):
            return self.sigma_min
        return torch.full_like(t, self.sigma_min)
    
    def sample_flow_matching_trajectory(self, theta_1: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample trajectory points for flow matching: θ_t = (1-t)θ_0 + tθ_1 + σ(t)ε
        
        Args:
            theta_1: Target parameters [batch_size, theta_dim]
            x: Observations [batch_size, x_dim]
            t: Time values [batch_size, 1] or [batch_size]
            
        Returns:
            Tuple of (theta_0, theta_t) where theta_0 ~ N(0,I) and theta_t follows flow matching
        """
        batch_size = theta_1.shape[0]
        
        # Ensure t is properly shaped
        if t.dim() == 1:
            t = t.view(-1, 1)
        elif t.dim() == 0:
            t = t.expand(batch_size, 1)
            
        # Sample noise (prior)
        theta_0 = torch.randn_like(theta_1)
        epsilon = torch.randn_like(theta_1)
        
        # Flow matching trajectory: θ_t = (1-t)θ_0 + tθ_1 + σ(t)ε
        sigma_t = self.flow_matching_scheduler(t)
        if sigma_t.dim() == 1:
            sigma_t = sigma_t.view(-1, 1)
            
        theta_t = (1 - t) * theta_0 + t * theta_1 + sigma_t * epsilon
        
        return theta_0, theta_t
    
    def generate_flow_matching_data(self, theta_batch: torch.Tensor, x_batch: torch.Tensor, n_samples_per_pair: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Koopman training data using direct flow matching (no teacher model)
        
        Args:
            theta_batch: Parameter samples from training data [batch_size, theta_dim]
            x_batch: Corresponding observations [batch_size, x_dim]
            n_samples_per_pair: Number of time samples per (theta, x) pair
            
        Returns:
            Tuple of (theta_0, theta_1, x_repeated, t_samples)
        """
        batch_size = theta_batch.shape[0]
        total_samples = batch_size * n_samples_per_pair
        
        # Repeat data for multiple time samples
        theta_1_repeated = theta_batch.repeat_interleave(n_samples_per_pair, dim=0)
        x_repeated = x_batch.repeat_interleave(n_samples_per_pair, dim=0)
        
        # Sample random times
        t_samples = torch.rand(total_samples, 1).to(self.device)
        
        # Generate trajectory points
        theta_0, theta_t = self.sample_flow_matching_trajectory(theta_1_repeated, x_repeated, t_samples)
        
        return theta_0, theta_1_repeated, x_repeated, t_samples
    
    def initialize_buffer(self, x_samples: torch.Tensor = None, theta_samples: torch.Tensor = None, min_samples: int = None):
        """
        Initialize buffer with training data (supports both teacher and direct modes)
        
        Args:
            x_samples: Observation samples 
            theta_samples: Parameter samples (required for direct mode)
            min_samples: Not used - buffer size matches available data
        """
        if not self.teacher_mode:
            # Direct flow matching mode - use training data
            assert theta_samples is not None and x_samples is not None, "Both theta and x samples required for direct mode"
            logger.info("Using direct flow matching for buffer initialization")
            
            n_available = min(len(theta_samples), len(x_samples))
            n_time_samples = 3  # Multiple time points per data pair
            
            for i in range(n_available):
                # Generate multiple time samples for this data pair
                theta_1 = theta_samples[i].unsqueeze(0).to(self.device)
                x = x_samples[i].unsqueeze(0).to(self.device)
                
                # Sample random times and generate trajectory points
                t_samples = torch.rand(n_time_samples, 1).to(self.device)
                theta_0_batch, theta_t_batch = self.sample_flow_matching_trajectory(
                    theta_1.repeat(n_time_samples, 1), 
                    x.repeat(n_time_samples, 1), 
                    t_samples
                )
                
                # Add all time samples to buffer (use theta_0 and theta_1 for consistency)
                for j in range(n_time_samples):
                    self.buffer.append((theta_0_batch[j].cpu(), theta_1.squeeze(0).cpu(), x.squeeze(0).cpu()))
                    
            logger.info(f"Buffer initialized with {len(self.buffer)} flow matching samples")
            
        elif hasattr(self, 'buffer_source') and self.buffer_source == 'data' and theta_samples is not None:
            # Legacy: Use ALL training data pairs 
            logger.info("Using all training data for buffer initialization")
            n_available = min(len(theta_samples), len(x_samples))
            
            for i in range(n_available):
                context = x_samples[i]                   # observation context
                x0 = torch.randn(theta_samples[i].shape).to(self.device)  # x0 ~ prior (noise)
                x1 = theta_samples[i].to(self.device)    # x1 ~ data
                self.buffer.append((x0, x1, context))
                
            logger.info(f"Buffer initialized with ALL {len(self.buffer)} training samples")
                
        else:
            # Teacher model sampling - match training set size, process in batches
            logger.info("Using teacher model sampling for buffer initialization")
            assert x_samples is not None, "x_samples must be provided for teacher sampling"
            assert self.teacher_model is not None, "Teacher model must be loaded"
            
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
    
    def get_koopman_operator(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Construct conditional Koopman operator L(x) with block-diagonal structure
        
        The operator parameters now depend on the observation x.
        
        Args:
            x: Observation tensor [batch_size, context_dim]
            dt: Time step (default 1.0 for t=0 to t=1)
            
        Returns:
            Koopman operator matrices [batch_size, lifted_dim, lifted_dim]
        """
        if not self.use_matrix_exponential:
            raise RuntimeError("Matrix exponential methods only available when use_matrix_exponential=True")
            
        batch_size = x.shape[0]
        L = torch.zeros(batch_size, self.lifted_dim, self.lifted_dim).to(self.device)
        
        # Generate conditional parameters from observation x
        alpha_params = self.alpha_net(x)  # [batch_size, n_blocks]
        beta_params = self.beta_net(x)    # [batch_size, n_blocks]
        Agt_params = self.Agt_net(x)      # [batch_size, internal_dim] 
        bg_params = self.bg_net(x)        # [batch_size, internal_dim]
        
        # Construct conditional block-diagonal A_gg matrix
        A_gg = self.construct_conditional_block_diagonal_matrix(alpha_params, beta_params)
        
        # L has structure:
        # [0, 0, 0_g^T]      <- row 0: [1] coordinate (unchanged)
        # [0, 0, A_gt^T]     <- row 1: [t] coordinate (unchanged) 
        # [0, b_g, A_gg]     <- rows 2+: internal coordinates
        
        # Bottom-right block: A_gg (conditional block-diagonal)
        L[:, 2:, 2:] = A_gg
        
        # Second column: A_gt coupling (conditional)
        L[:, 2:, 1:2] = Agt_params.unsqueeze(-1)
        
        # First column: b_g bias (conditional)
        L[:, 2:, 0:1] = bg_params.unsqueeze(-1)
        
        return L * dt
    
    def construct_conditional_block_diagonal_matrix(self, alpha_params: torch.Tensor, beta_params: torch.Tensor) -> torch.Tensor:
        """
        Construct conditional block-diagonal matrix A_gg from α and β parameters
        
        Each 2x2 block has the form:
        [α_k  β_k ]
        [-β_k α_k ]
        
        This corresponds to complex eigenvalues α_k ± i*β_k
        
        Args:
            alpha_params: Alpha parameters [batch_size, n_blocks]
            beta_params: Beta parameters [batch_size, n_blocks]
        
        Returns:
            Block-diagonal matrices [batch_size, internal_dim, internal_dim]
        """
        batch_size = alpha_params.shape[0]
        A_gg = torch.zeros(batch_size, self.internal_dim, self.internal_dim).to(self.device)
        
        for k in range(self.n_blocks):
            # Get parameters for this block across all batches
            alpha_k = alpha_params[:, k]  # [batch_size]
            beta_k = beta_params[:, k]    # [batch_size]
            
            # Block indices
            i = 2 * k
            j = 2 * k + 1
            
            # Fill 2x2 block for all batches
            A_gg[:, i, i] = alpha_k      # Top-left
            A_gg[:, i, j] = beta_k       # Top-right  
            A_gg[:, j, i] = -beta_k      # Bottom-left
            A_gg[:, j, j] = alpha_k      # Bottom-right
            
        return A_gg
    
    def matrix_exp_batch(self, L: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix exponential exp(L) exploiting block-diagonal structure
        
        This is much faster than general matrix exponential since:
        1. exp(block_diag(A1, A2, ...)) = block_diag(exp(A1), exp(A2), ...)
        2. Each 2x2 block has analytical exponential
        3. The [1,t] coordinates are handled separately
        
        Args:
            L: Koopman operator matrices [batch_size, lifted_dim, lifted_dim]
            
        Returns:
            Matrix exponentials [batch_size, lifted_dim, lifted_dim]
        """
        if not self.use_matrix_exponential:
            raise RuntimeError("Matrix exponential methods only available when use_matrix_exponential=True")
            
        batch_size, n, _ = L.shape
        device = L.device
        
        # Initialize result as identity
        result = torch.eye(n, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Handle the first two coordinates [1, t] - they're not affected by the generator
        # result[0,0] = 1 (already set by identity)
        # result[1,1] = 1 (already set by identity)
        
        # Handle time evolution: the [1] coordinate couples to [t] coordinate
        # Since L[0, :] = [0, 0, 0, ...], we have exp(L)[0, :] = [1, 0, 0, ...]
        # Since L[1, :] = [0, 0, A_gt[0], A_gt[1], ...], we get non-trivial coupling
        
        # For the simplified structure where only internal coordinates evolve:
        # exp(L)[0, 0] = 1, exp(L)[0, j] = 0 for j > 0
        # exp(L)[1, 1] = 1, exp(L)[1, j] = L[1, j] for j > 1 (first-order)
        
        # Copy the time coupling (first-order approximation since dt=1)
        result[:, 1, 2:] = L[:, 1, 2:]  # A_gt^T coupling
        
        # Handle bias terms (first-order approximation)
        result[:, 2:, 0] = L[:, 2:, 0].squeeze(-1)  # b_g bias
        
        # Now handle the internal coordinates block-diagonal matrix exponential
        # Extract A_gg block (internal coordinates)
        A_gg = L[:, 2:, 2:]  # [batch_size, internal_dim, internal_dim]
        
        # Compute exp(A_gg) using block-diagonal structure
        exp_A_gg = self.block_diagonal_matrix_exp(A_gg)
        
        # Insert back into result
        result[:, 2:, 2:] = exp_A_gg
        
        return result
    
    def block_diagonal_matrix_exp(self, A_gg: torch.Tensor) -> torch.Tensor:
        """
        Compute exponential of block-diagonal matrix analytically
        
        For each 2x2 block [α β; -β α], the exponential is:
        exp([α β; -β α]) = exp(α) * [cos(β) sin(β); -sin(β) cos(β)]
        
        Args:
            A_gg: Block-diagonal matrices [batch_size, internal_dim, internal_dim]
            
        Returns:
            Matrix exponentials [batch_size, internal_dim, internal_dim]
        """
        batch_size, internal_dim, _ = A_gg.shape
        device = A_gg.device
        
        # Initialize result
        result = torch.zeros_like(A_gg)
        
        # Process each 2x2 block
        for k in range(self.n_blocks):
            # Block indices
            i = 2 * k
            j = 2 * k + 1
            
            # Extract α and β for all batches
            alpha = A_gg[:, i, i]  # Should be same as A_gg[:, j, j]
            beta = A_gg[:, i, j]   # Should be -A_gg[:, j, i]
            
            # Compute exp(α), cos(β), sin(β)
            exp_alpha = torch.exp(alpha)
            cos_beta = torch.cos(beta)
            sin_beta = torch.sin(beta)
            
            # Fill the 2x2 block: exp(α) * [cos(β) sin(β); -sin(β) cos(β)]
            result[:, i, i] = exp_alpha * cos_beta    # Top-left
            result[:, i, j] = exp_alpha * sin_beta    # Top-right
            result[:, j, i] = -exp_alpha * sin_beta   # Bottom-left  
            result[:, j, j] = exp_alpha * cos_beta    # Bottom-right
            
        return result
    
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
            
        # Choose data source based on mode
        if hasattr(self, 'theta_0') and self.dataset_size > 0:
            # Use pre-generated dataset if available
            batch_size = min(self.dataset_size, theta.shape[0])
            dataset_indices = np.random.choice(self.dataset_size, size=batch_size, replace=False)
            
            theta_0_buffer = self.theta_0[dataset_indices]
            theta_1_buffer = self.theta_1[dataset_indices]
            x_buffer = self.x_context[dataset_indices]
            
        elif not self.teacher_mode:
            # Direct flow matching mode - generate data on the fly
            batch_size = theta.shape[0]
            
            # Use current batch as theta_1 and generate theta_0 via flow matching
            theta_0_buffer, theta_1_buffer, x_buffer, _ = self.generate_flow_matching_data(
                theta, x, n_samples_per_pair=1
            )
            
        elif len(self.buffer) > 0:
            # Use buffer (teacher mode legacy)
            batch_size = min(len(self.buffer), theta.shape[0])
            buffer_indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            
            theta_0_buffer = torch.stack([self.buffer[i][0] for i in buffer_indices]).to(self.device)
            theta_1_buffer = torch.stack([self.buffer[i][1] for i in buffer_indices]).to(self.device)
            x_buffer = torch.stack([self.buffer[i][2] for i in buffer_indices]).to(self.device)
            
        else:
            raise RuntimeError("No training data available. Either load pre-generated dataset or initialize buffer.")
        
        # 1. PHASE LOSS: ||e^L(x) * gφ(0, θ₀) - gφ(1, θ₁)||²
        z_0 = self.compute_lifted_coords(torch.zeros(len(theta_0_buffer), 1).to(self.device), 
                                        theta_0_buffer, x_buffer)
        z_1_target = self.compute_lifted_coords(torch.ones(len(theta_1_buffer), 1).to(self.device),
                                               theta_1_buffer, x_buffer)
        
        # Evolution based on selected method
        if self.use_matrix_exponential:
            # Matrix exponential evolution (conditional on x)
            L_conditional = self.get_koopman_operator(x_buffer, dt=1.0)
            exp_L = self.matrix_exp_batch(L_conditional)
            z_1_evolved = torch.bmm(exp_L, z_0.unsqueeze(-1)).squeeze(-1)
        else:
            # Linear evolution (without additive control for now)
            z_1_evolved = self.linear_evolution(z_0)
        
        L_phase = torch.mean((z_1_evolved - z_1_target)**2)
        
        # 2. TARGET LOSS: ||decoder(e^L * gφ(0, θ₀)) - θ₁||²
        theta_1_decoded = self.decoder(z_1_evolved)
        L_target = torch.mean((theta_1_decoded - theta_1_buffer)**2)
        
        # 3. RECONSTRUCTION LOSS: ||decoder(gφ(1, θ₁)) - θ₁||²
        theta_1_recon = self.decoder(z_1_target)
        L_recon = torch.mean((theta_1_recon - theta_1_buffer)**2)
        
        # 4. CONSISTENCY LOSS (skip if lambda_cons = 0.0)
        if self.lambda_cons > 0.0:
            if self.teacher_mode and self.teacher_model is not None:
                # Teacher model consistency loss
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
                # Direct flow matching mode - use analytical vector field
                # v(t, θ_t, x) = θ_1 - θ_0 (constant velocity for optimal transport)
                # For now, skip consistency loss in direct mode since we don't have vector field
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
        
        # Choose data source based on mode (same as training loss)
        if hasattr(self, 'theta_0') and self.dataset_size > 0:
            # Use pre-generated dataset if available
            batch_size = min(self.dataset_size, theta.shape[0])
            dataset_indices = np.random.choice(self.dataset_size, size=batch_size, replace=False)
            
            theta_0_buffer = self.theta_0[dataset_indices]
            theta_1_buffer = self.theta_1[dataset_indices]
            x_buffer = self.x_context[dataset_indices]
            
        elif not self.teacher_mode:
            # Direct flow matching mode - generate data on the fly
            batch_size = theta.shape[0]
            
            # Use current batch as theta_1 and generate theta_0 via flow matching
            theta_0_buffer, theta_1_buffer, x_buffer, _ = self.generate_flow_matching_data(
                theta, x, n_samples_per_pair=1
            )
            
        elif len(self.buffer) > 0:
            # Use buffer (teacher mode legacy)
            batch_size = min(len(self.buffer), theta.shape[0])
            buffer_indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            
            theta_0_buffer = torch.stack([self.buffer[i][0] for i in buffer_indices]).to(self.device)
            theta_1_buffer = torch.stack([self.buffer[i][1] for i in buffer_indices]).to(self.device)
            x_buffer = torch.stack([self.buffer[i][2] for i in buffer_indices]).to(self.device)
            
        else:
            # Return simple reconstruction loss if no data available
            return torch.mean((theta - theta)**2)  # Dummy loss
        
        # Compute only core Koopman losses (same as training but no buffer updates)
        # 1. PHASE LOSS with conditional control
        z_0 = self.compute_lifted_coords(torch.zeros(len(theta_0_buffer), 1).to(self.device), 
                                        theta_0_buffer, x_buffer)
        z_1_target = self.compute_lifted_coords(torch.ones(len(theta_1_buffer), 1).to(self.device),
                                               theta_1_buffer, x_buffer)
        
        # Evolution based on selected method
        if self.use_matrix_exponential:
            # Matrix exponential evolution (conditional on x)
            L_conditional = self.get_koopman_operator(x_buffer, dt=1.0)
            exp_L = self.matrix_exp_batch(L_conditional)
            z_1_evolved = torch.bmm(exp_L, z_0.unsqueeze(-1)).squeeze(-1)
        else:
            # Linear evolution (without additive control for now)
            z_1_evolved = self.linear_evolution(z_0)
        
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
        
        # Sample noise (prior) - compatible with both modes
        if self.teacher_mode and hasattr(self.teacher_model, 'sample_theta_0'):
            theta_0 = self.teacher_model.sample_theta_0(batch_size).to(self.device)
        else:
            # Use standard Gaussian prior
            theta_0 = torch.randn(batch_size, self.input_dim).to(self.device)
        
        # Lift to Koopman space at t=0
        z_0 = self.compute_lifted_coords(
            torch.zeros(batch_size, 1).to(self.device),
            theta_0,
            x
        )
        
        # One-step evolution to t=1 based on selected method
        if self.use_matrix_exponential:
            # Matrix exponential evolution (conditional on x)
            L_conditional = self.get_koopman_operator(x, dt=1.0)
            exp_L = self.matrix_exp_batch(L_conditional)
            z_1 = torch.bmm(exp_L, z_0.unsqueeze(-1)).squeeze(-1)
        else:
            # Linear evolution (without additive control for now)
            z_1 = self.linear_evolution(z_0)
        
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
        
        # Evolution parameters based on method
        if self.use_matrix_exponential:
            # Conditional networks for matrix exponential parameters
            if hasattr(self, 'alpha_net'):
                params.extend(self.alpha_net.parameters())
            if hasattr(self, 'beta_net'):
                params.extend(self.beta_net.parameters())
            if hasattr(self, 'Agt_net'):
                params.extend(self.Agt_net.parameters())
            if hasattr(self, 'bg_net'):
                params.extend(self.bg_net.parameters())
        else:
            # Linear evolution parameters
            if hasattr(self, 'linear_evolution'):
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
        
        # Evolution parameters based on method
        if self.use_matrix_exponential:
            # Conditional networks for matrix exponential parameters
            if hasattr(self, 'alpha_net'):
                state['alpha_net'] = self.alpha_net.state_dict()
            if hasattr(self, 'beta_net'):
                state['beta_net'] = self.beta_net.state_dict()
            if hasattr(self, 'Agt_net'):
                state['Agt_net'] = self.Agt_net.state_dict()
            if hasattr(self, 'bg_net'):
                state['bg_net'] = self.bg_net.state_dict()
        else:
            # Linear evolution parameters
            if hasattr(self, 'linear_evolution'):
                state['linear_evolution'] = self.linear_evolution.state_dict()
        
        # Conditional control network
        if hasattr(self, 'control_linear'):
            state['control_linear'] = self.control_linear.state_dict()
        
        # Store method flag
        state['use_matrix_exponential'] = self.use_matrix_exponential
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dictionary into model components"""
        if 'encoder' in state_dict and hasattr(self, 'encoder'):
            self.encoder.load_state_dict(state_dict['encoder'])
        if 'decoder' in state_dict and hasattr(self, 'decoder'):
            self.decoder.load_state_dict(state_dict['decoder'])
        
        # Load method flag if available
        if 'use_matrix_exponential' in state_dict:
            self.use_matrix_exponential = state_dict['use_matrix_exponential']
        
        # Evolution parameters based on method
        if self.use_matrix_exponential:
            # Conditional networks for matrix exponential parameters
            if 'alpha_net' in state_dict and hasattr(self, 'alpha_net'):
                self.alpha_net.load_state_dict(state_dict['alpha_net'])
            if 'beta_net' in state_dict and hasattr(self, 'beta_net'):
                self.beta_net.load_state_dict(state_dict['beta_net'])
            if 'Agt_net' in state_dict and hasattr(self, 'Agt_net'):
                self.Agt_net.load_state_dict(state_dict['Agt_net'])
            if 'bg_net' in state_dict and hasattr(self, 'bg_net'):
                self.bg_net.load_state_dict(state_dict['bg_net'])
        else:
            # Linear evolution parameters
            if 'linear_evolution' in state_dict and hasattr(self, 'linear_evolution'):
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
        if hasattr(self, 'linear_evolution'):
            self.linear_evolution.eval()
        # Conditional networks for matrix exponential
        if hasattr(self, 'alpha_net'):
            self.alpha_net.eval()
        if hasattr(self, 'beta_net'):
            self.beta_net.eval()
        if hasattr(self, 'Agt_net'):
            self.Agt_net.eval()
        if hasattr(self, 'bg_net'):
            self.bg_net.eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode"""
        if hasattr(self, 'encoder'):
            self.encoder.train(mode)
        if hasattr(self, 'decoder'):
            self.decoder.train(mode)
        if hasattr(self, 'linear_evolution'):
            self.linear_evolution.train(mode)
        # Conditional networks for matrix exponential
        if hasattr(self, 'alpha_net'):
            self.alpha_net.train(mode)
        if hasattr(self, 'beta_net'):
            self.beta_net.train(mode)
        if hasattr(self, 'Agt_net'):
            self.Agt_net.train(mode)
        if hasattr(self, 'bg_net'):
            self.bg_net.train(mode)
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
    
    def run_training(self, train_loader, test_loader, train_dir, runtime_limits, early_stopping=True, use_wandb=False):
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
                        'use_matrix_exponential': self.use_matrix_exponential,
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