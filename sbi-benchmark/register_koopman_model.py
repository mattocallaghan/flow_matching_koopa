"""
Model Registration Helper for Koopman SBI Model

This script helps register the KoopmanSBIModel with the dingo build system
so it can be used with autocomplete_model_kwargs and build_model_from_kwargs.

Usage:
    # Import this before using KoopmanSBIModel
    from register_koopman_model import register_koopman_model
    register_koopman_model()
    
    # Then use normally
    from dingo.core.posterior_models.build_model import build_model_from_kwargs
"""

import logging

logger = logging.getLogger(__name__)


def register_koopman_model():
    """Register KoopmanSBIModel with dingo's build system"""
    
    try:
        # Import dingo components
        from dingo.core.posterior_models import build_model
        from koopman_sbi_model import KoopmanSBIModel
        
        # Add to model registry
        if hasattr(build_model, '_MODEL_DICT'):
            build_model._MODEL_DICT['koopman_sbi'] = KoopmanSBIModel
            logger.info("Registered KoopmanSBIModel as 'koopman_sbi'")
        else:
            # Fallback for different dingo versions
            if not hasattr(build_model, 'KOOPMAN_REGISTERED'):
                # Monkey patch the get_model_class function
                original_get_model_class = getattr(build_model, 'get_model_class', None)
                
                def patched_get_model_class(model_type, **kwargs):
                    if model_type == 'koopman_sbi':
                        return KoopmanSBIModel
                    elif original_get_model_class:
                        return original_get_model_class(model_type, **kwargs)
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                
                build_model.get_model_class = patched_get_model_class
                build_model.KOOPMAN_REGISTERED = True
                logger.info("Registered KoopmanSBIModel via monkey patching")
                
    except ImportError:
        logger.warning("Dingo not available, model registration skipped")
    except Exception as e:
        logger.error(f"Failed to register KoopmanSBIModel: {e}")


def create_koopman_sbi_model(settings_dict: dict, device: str = 'cpu'):
    """
    Factory function to create KoopmanSBIModel with proper settings
    
    Args:
        settings_dict: Dictionary containing model configuration
        device: Device to run on
        
    Returns:
        Initialized KoopmanSBIModel
    """
    from koopman_sbi_model import KoopmanSBIModel
    
    # Extract model configuration
    model_config = settings_dict.get('model', {})
    
    # Extract dimensions (will be set by autocomplete_model_kwargs)
    input_dim = model_config.get('input_dim', 2)
    context_dim = model_config.get('context_dim', 2)
    
    # Koopman-specific parameters
    koopman_kwargs = {
        'teacher_model_path': model_config.get('teacher_model_path'),
        'lifted_dim': model_config.get('lifted_dim', 64),
        'lambda_phase': model_config.get('lambda_phase', 1.0),
        'lambda_target': model_config.get('lambda_target', 1.0),
        'lambda_recon': model_config.get('lambda_recon', 1.0),
        'lambda_cons': model_config.get('lambda_cons', 0.1),
        'buffer_size': model_config.get('buffer_size', 5000)
    }
    
    # Standard dingo parameters
    dingo_kwargs = {
        'input_dim': input_dim,
        'context_dim': context_dim,
        'device': device
    }
    
    # Combine all parameters
    all_kwargs = {**dingo_kwargs, **koopman_kwargs}
    
    # Remove None values
    all_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
    
    # Create model
    model = KoopmanSBIModel(**all_kwargs)
    # Model networks are already on device from initialization
    model.initialize_network()
    
    return model


# Register automatically when imported
register_koopman_model()