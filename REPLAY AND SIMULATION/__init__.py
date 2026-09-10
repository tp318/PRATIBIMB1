"""
============================================================================
REPLAY AND SIMULATION — Complete Portable Sortie Replay & Mission Simulation
============================================================================
Provides:
  - Flight trajectory replayer (streaming at 50 Hz from CSV or synthetic)
  - Atmospheric environmental model (ISA lapse rate, 0 to 7600m)
  - Mission flight clearance engine (GO / CAUTION_GO / NO-GO)
  - Mission scenario presets & throttle drivers (High Alt, Endurance, Hot Weather)
  - High-throughput mission persistence & replay (PostgreSQL + SQLite)
"""

try:
    from .flight_replay import FlightReplayer, FlightSample, FAULT_NAMES
    from .env_simulator import EnvironmentalSimulator, AtmosphericState
    from .mission_engine import (
        MissionSimulator,
        MissionProfile,
        ClearanceStatus,
        MissionAssessment,
    )
    from .mission_scenario import (
        ScenarioConfig,
        ScenarioType,
        ThrottleProfile,
        build_scenario,
        list_presets,
        PRESETS,
    )
    from .database import (
        PostgresMissionDatabase,
        SqliteMissionDatabase,
        get_mission_database,
        get_database,
    )
except ImportError:
    from flight_replay import FlightReplayer, FlightSample, FAULT_NAMES
    from env_simulator import EnvironmentalSimulator, AtmosphericState
    from mission_engine import (
        MissionSimulator,
        MissionProfile,
        ClearanceStatus,
        MissionAssessment,
    )
    from mission_scenario import (
        ScenarioConfig,
        ScenarioType,
        ThrottleProfile,
        build_scenario,
        list_presets,
        PRESETS,
    )
    from database import (
        PostgresMissionDatabase,
        SqliteMissionDatabase,
        get_mission_database,
        get_database,
    )

__all__ = [
    # Flight Replay
    "FlightReplayer",
    "FlightSample",
    "FAULT_NAMES",
    # Environment
    "EnvironmentalSimulator",
    "AtmosphericState",
    # Mission Clearance
    "MissionSimulator",
    "MissionProfile",
    "ClearanceStatus",
    "MissionAssessment",
    # Scenario Presets & Profiles
    "ScenarioConfig",
    "ScenarioType",
    "ThrottleProfile",
    "build_scenario",
    "list_presets",
    "PRESETS",
    # Database Persistence & Replay
    "PostgresMissionDatabase",
    "SqliteMissionDatabase",
    "get_mission_database",
    "get_database",
]
