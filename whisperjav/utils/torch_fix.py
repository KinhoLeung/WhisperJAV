import torch.hub
import logging
from pathlib import Path
from typing import Optional
import os

logger = logging.getLogger("whisperjav")

def apply_torch_hub_fix():
    """
    Apply a global monkey-patch to torch.hub to skip the 'untrusted repository' check.
    
    This prevents EOFError in non-interactive environments like Google Colab 
    where torch.hub tries to call input() to ask for trust.
    """
    try:
        if not hasattr(torch.hub, '_check_repo_is_trusted_original'):
            # Save the original function just in case
            torch.hub._check_repo_is_trusted_original = torch.hub._check_repo_is_trusted
            
            # Replace with a function that always returns True
            torch.hub._check_repo_is_trusted = lambda *args, **kwargs: True
            logger.debug("Applied global torch.hub trust patch to bypass interactive EOFError")
    except Exception as e:
        logger.warning(f"Failed to apply torch.hub trust patch: {e}")

def get_silero_vad_local_path(config_path: Optional[str] = None) -> Optional[str]:
    """
    Resolve the local path for Silero VAD based on config or environment variable.
    
    Order of precedence:
    1. config_path (from VAD options)
    2. SILERO_VAD_LOCAL_PATH environment variable
    """
    # 1. Check config path
    if config_path:
        p = Path(config_path).expanduser()
        if p.exists() and p.is_dir():
            return str(p)
        logger.warning(f"Configured Silero VAD local path does not exist or is not a directory: {config_path}")

    # 2. Check environment variable
    env_path = os.environ.get("SILERO_VAD_LOCAL_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists() and p.is_dir():
            return str(p)
        logger.warning(f"SILERO_VAD_LOCAL_PATH environment variable points to invalid path: {env_path}")

    return None
