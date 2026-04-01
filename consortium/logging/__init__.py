"""
Logging infrastructure for the consortium multi-agent system.
"""
from .training_data_logger import (
    initialize_training_data_logger,
    get_training_data_logger,
    set_current_agent_name,
    get_current_agent_name,
)