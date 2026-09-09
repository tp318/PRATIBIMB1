"""
AeroTwin-4 Physical Degradation Mechanisms.

Maps component severity levels S in [0.0, 1.0] to physical engine subsystem parameters:
- D1 Cylinder: Combustion efficiency E_comb = 1.0 - 0.50 * S
- D2 Bearing: Bearing mechanical friction multiplier M_bearing = 1.0 + 1.00 * S
- D3 Cooling: Cooling heat dissipation efficiency E_cool = 1.0 - 0.50 * S
- D4 Lubrication: Oil system pump efficiency E_lub = 1.0 - 0.50 * S

Note:
These mappings represent prototype simulator assumptions and are explicitly labeled as such.
"""

from typing import Dict, Any, List
from .config import ComponentDegradation, DegradationType, ComponentID


class PhysicalDegradationMapper:
    """
    Translates high-level component degradation severities into low-level
    physics subsystem parameters consumed by EngineDynamics.
    """

    @staticmethod
    def map_degradations_to_physics_params(
        degradations: List[ComponentDegradation]
    ) -> Dict[str, Any]:
        """
        Calculate physical subsystem parameters from active component degradations.

        Returns
        -------
        dict
            Dict containing:
            - combustion_efficiencies: Dict[int, float]
            - bearing_friction_multiplier: float
            - cooling_efficiency: float
            - lubrication_efficiency: float
        """
        combustion_efficiencies = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        bearing_friction_mult = 1.0
        cooling_eff = 1.0
        lubrication_eff = 1.0

        for deg in degradations:
            s = max(0.0, min(1.0, float(deg.severity)))

            if deg.degradation_type == DegradationType.CYLINDER:
                # Map cylinder ID (e.g. CYLINDER_3 -> 3)
                cyl_map = {
                    ComponentID.CYLINDER_1: 1,
                    ComponentID.CYLINDER_2: 2,
                    ComponentID.CYLINDER_3: 3,
                    ComponentID.CYLINDER_4: 4,
                }
                cyl_num = cyl_map.get(deg.component_id)
                if cyl_num:
                    combustion_efficiencies[cyl_num] = max(0.50, 1.0 - 0.50 * s)

            elif deg.degradation_type == DegradationType.BEARING:
                # Mechanical dry/boundary friction increase
                bearing_friction_mult = max(1.0, 1.0 + 1.00 * s)

            elif deg.degradation_type == DegradationType.COOLING:
                # Heat dissipation efficiency reduction
                cooling_eff = max(0.50, 1.0 - 0.50 * s)

            elif deg.degradation_type == DegradationType.LUBRICATION:
                # Oil pump efficiency / pressure capacity reduction
                lubrication_eff = max(0.50, 1.0 - 0.50 * s)

        return {
            "combustion_efficiencies": combustion_efficiencies,
            "bearing_friction_multiplier": bearing_friction_mult,
            "cooling_efficiency": cooling_eff,
            "lubrication_efficiency": lubrication_eff,
        }
