# =============================================================================
# Packages Setup
# =============================================================================
import logging
from paths import LOG_DIR

# Configure centralized logging for pipeline monitoring and debugging.
logging.basicConfig(
    filename=LOG_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True
)