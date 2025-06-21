"""
AI Pause Model - Parameters Configuration
Extracted from Playbook modeling 1.xlsx
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class DistributionParams:
    """Holds 15th, 50th, 85th percentiles for a parameter"""
    p15: float
    p50: float  
    p85: float
    
    def to_lognormal_params(self):
        """Convert percentiles to lognormal distribution parameters (mu, sigma)"""
        # Using the 50th percentile as the median and deriving sigma from the spread
        median = self.p50
        # Calculate sigma using the relationship between percentiles
        # For lognormal: ln(p85/p50) ≈ 1.036*sigma, ln(p50/p15) ≈ 1.036*sigma
        upper_ratio = np.log(self.p85 / self.p50) / 1.036
        lower_ratio = np.log(self.p50 / self.p15) / 1.036
        sigma = (upper_ratio + lower_ratio) / 2
        mu = np.log(median)
        return mu, sigma

# Global Hardware Parameters (March 2027 starting conditions)
HARDWARE_GLOBAL = {
    'compute_million_h100e': DistributionParams(20, 50, 100),
    'memory_million_tb': DistributionParams(1.4, 3.5, 7.0),
    'bandwidth_million_tb_s': DistributionParams(28, 70, 140),
}

# Hardware shares by actor type (fixing 0 values)
HARDWARE_SHARES = {
    'coalition': DistributionParams(0.8, 0.9, 0.98),
    'all_criminal_orgs': DistributionParams(0.002, 0.01, 0.02),
    'us_secret_black_sites': DistributionParams(0.004, 0.02, 0.04),
    'china_secret_black_sites': DistributionParams(0.004, 0.02, 0.04),
    'eu_secret_black_sites': DistributionParams(0.002, 0.01, 0.02),
    'random_other_black_sites': DistributionParams(0.0018, 0.009, 0.018),
    'russia': DistributionParams(0.002, 0.01, 0.02),
    'iran': DistributionParams(0.0001, 0.0005, 0.001),
    'other_non_coalition_nations': DistributionParams(0.001, 0.005, 0.01),
}

# Pre-deal hardware ownership
PRE_DEAL_OWNERSHIP = {
    'us_share': DistributionParams(0.5, 0.75, 0.85),
    'china_share': DistributionParams(0.05, 0.12, 0.25),
    'rest_of_world_share': DistributionParams(0.05, 0.13, 0.25),
    'non_agi_usage_share': DistributionParams(0.05, 0.1, 0.2),
}

# Cluster size distributions
CLUSTER_SIZES = {
    'coalition': {
        'large_cluster_share': 0.72,  # >100K H100e
        'medium_cluster_share': 0.23,  # 1K-100K H100e
        'small_cluster_share': 0.05,  # <1K H100e
    },
    'non_coalition': {
        'large_cluster_share': 0.12,
        'medium_cluster_share': 0.53,
        'small_cluster_share': 0.35,
    }
}

# Software/R&D Parameters
SOFTWARE_PARAMS = {
    'leading_internal_ai_rd_mult': DistributionParams(1.5, 4, 10),
}

# Months behind coalition by actor type
MONTHS_BEHIND = {
    'criminal_orgs': 'use_open_source',  # Special case: months behind open source
    'us_secret_black_sites': DistributionParams(0.001, 0.5, 3),
    'china_secret_black_sites': DistributionParams(0.001, 1, 4),
    'eu_secret_black_sites': DistributionParams(0.001, 2, 6),
    'random_other_black_sites': 'use_open_source',  # Special case: months behind open source
    'other_nation_states': 'use_open_source',  # Special case: months behind open source
}

# Reference months behind values
REFERENCE_MONTHS_BEHIND = {
    'us_2nd_place': DistributionParams(0.1, 2, 5),
    'us_3rd_place': DistributionParams(0.5, 3, 8),
    'china_1st_place': DistributionParams(1, 4, 12),
    'open_source': DistributionParams(1.5, 6, 18),
}

# AI Research Talent (1.0 = 1x OpenAI 2024 equivalent)
AI_RESEARCH_TALENT = {
    'coalition': DistributionParams(3, 8, 25),
    'criminal_orgs': DistributionParams(0.05, 0.3, 1.5),
    'us_secret_black_sites': DistributionParams(0.5, 1.5, 4),
    'china_secret_black_sites': DistributionParams(0.5, 2, 10),
    'eu_secret_black_sites': DistributionParams(0.2, 0.8, 3),
    'random_other_black_sites': DistributionParams(0.05, 0.3, 1.5),
}

# Hardware lifetime distributions (in years)
HARDWARE_LIFETIME = {
    'compute_lifetime_years': DistributionParams(0.6, 1.8, 4.5),  # Lognormal with mean ~3 years
    'memory_lifetime_years': DistributionParams(0.6, 1.8, 4.5),   # Same distribution
    'bandwidth_lifetime_years': DistributionParams(0.6, 1.8, 4.5), # Same distribution
}

# Hardware scaling factors (for translating compute to memory/bandwidth)
HARDWARE_SCALING = {
    'memory_per_h100e_tb': 0.07,  # 70 GB per H100e
    'bandwidth_per_h100e_tb_s': 1.4,  # 1.4 TB/s per H100e
}

# Actor definitions and counts
ACTOR_DEFINITIONS = {
    'coalition_members': ['us', 'china', 'eu', 'rest_of_coalition'],
    'rogue_black_sites': {
        'us_rogue': 3,
        'china_rogue': 3, 
        'eu_rogue': 10,
        'other_coalition_rogue': 20  # Mix of sanctioned and unsanctioned
    },
    'criminal_orgs': {
        'major_criminal_orgs': 10,  # #1 through #10
        'minor_criminal_orgs': 25   # #11 through #35
    },
    'non_coalition_nations': [
        'russia', 'north_korea', 'iran', 'iraq', 'syria'
    ],
    'other_nations': 15  # Additional non-coalition nations
}

# Collaboration likelihood and penalty matrix
# Format: (actor1, actor2): {'likelihood': float, 'penalty': float}
# Likelihood: 0.0-1.0 probability of collaboration
# Penalty: 0.0-1.0 efficiency loss when collaborating (0 = no penalty, 1 = total loss)
COLLABORATION_MATRIX = {
    # High cooperation pairs
    ('iran', 'iraq'): {'likelihood': 0.9, 'penalty': 0.2},
    ('russia', 'syria'): {'likelihood': 0.9, 'penalty': 0.2},
    ('russia', 'iran'): {'likelihood': 0.7, 'penalty': 0.3},
    ('north_korea', 'iran'): {'likelihood': 0.6, 'penalty': 0.4},
    ('russia', 'north_korea'): {'likelihood': 0.5, 'penalty': 0.4},
    
    # Zero/minimal cooperation
    ('china', 'us_black_sites'): {'likelihood': 0.0, 'penalty': 1.0},
    ('us', 'china_black_sites'): {'likelihood': 0.0, 'penalty': 1.0},
    ('coalition', 'criminal_org'): {'likelihood': 0.0, 'penalty': 1.0},
    
    # Criminal organization collaboration (high penalty due to coordination issues)
    ('criminal_org', 'criminal_org'): {'likelihood': 0.3, 'penalty': 0.5},
    ('criminal_org', 'black_site'): {'likelihood': 0.2, 'penalty': 0.6},
    ('criminal_org', 'nation_state'): {'likelihood': 0.1, 'penalty': 0.7},
    
    # Black site collaboration
    ('us_black_site', 'us_black_site'): {'likelihood': 0.4, 'penalty': 0.3},
    ('china_black_site', 'china_black_site'): {'likelihood': 0.5, 'penalty': 0.3},
    ('eu_black_site', 'eu_black_site'): {'likelihood': 0.3, 'penalty': 0.4},
    ('black_site', 'nation_state'): {'likelihood': 0.2, 'penalty': 0.5},
    
    # Nation state collaboration
    ('nation_state', 'nation_state'): {'likelihood': 0.3, 'penalty': 0.4},
    
    # Default fallback for unspecified pairs
    ('default', 'default'): {'likelihood': 0.1, 'penalty': 0.8}
}

# Threat level distributions (using 15th, 50th, 85th percentiles from original sheet)
# These will be converted to discrete threat levels OC2-OC5
THREAT_LEVEL_DISTRIBUTIONS = {
    'coalition': DistributionParams(5.0, 5.5, 6.0),  # Lower threat (OC2-OC3)
    'black_sites': DistributionParams(3.0, 4.0, 6.0),  # Higher threat (OC3-OC5)
    'criminal_orgs': DistributionParams(1.5, 3.5, 4.0),  # Variable threat (OC3-OC4)
    'nation_states': DistributionParams(3.5, 4.0, 4.0),  # High threat (OC3-OC5)
    'other_nations': DistributionParams(1.5, 2.5, 4.0),  # Variable threat (OC2-OC4)
}

# Security level distributions (1-5 scale, placeholders for future parameterization)
SECURITY_LEVEL_DISTRIBUTIONS = {
    'coalition': DistributionParams(4.0, 4.5, 5.0),  # High security
    'black_sites': DistributionParams(2.0, 3.0, 4.0),  # Variable security
    'criminal_orgs': DistributionParams(1.0, 2.0, 3.0),  # Lower security
    'nation_states': DistributionParams(2.0, 3.0, 4.0),  # Variable by nation
    'other_nations': DistributionParams(1.0, 2.0, 3.0),  # Generally lower
}

# Military defense level distributions (1-5 scale, placeholders)
MILITARY_DEFENSE_DISTRIBUTIONS = {
    'coalition': DistributionParams(4.0, 4.5, 5.0),  # Strong military defense
    'black_sites': DistributionParams(1.0, 2.0, 3.0),  # Hidden, less defended
    'criminal_orgs': DistributionParams(1.0, 1.5, 2.0),  # Minimal military defense
    'nation_states': DistributionParams(2.0, 3.0, 4.0),  # Variable by nation
    'other_nations': DistributionParams(1.0, 2.0, 3.0),  # Generally weaker
}

# Purchasing power distributions (in billions USD equivalent)
PURCHASING_POWER = {
    'coalition': DistributionParams(100, 500, 2000),
    'criminal_orgs_major': DistributionParams(0.1, 1, 10),
    'criminal_orgs_minor': DistributionParams(0.01, 0.1, 1),
    'russia': DistributionParams(0.5, 5, 50),
    'iran': DistributionParams(0.1, 1, 10),
    'black_sites': DistributionParams(0.05, 0.5, 5),
    'other_nations': DistributionParams(0.01, 0.5, 10),
}

# AI R&D progression lookup table (months behind -> multiplier)
AI_RD_PROGRESSION = {
    'months': [0, 4, 8, 12, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    'multipliers': [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3, 4, 5, 7, 10, 15, 25, 50, 100, 250, 2000]
}

def months_behind_to_multiplier(leading_multiplier: float, months_behind: float) -> float:
    """Convert months behind to AI R&D multiplier based on leading project level"""
    import numpy as np
    
    # Find current position of leading project
    multipliers = AI_RD_PROGRESSION['multipliers']
    months = AI_RD_PROGRESSION['months']
    
    # Find closest multiplier to leading project
    leading_idx = np.argmin([abs(m - leading_multiplier) for m in multipliers])
    leading_months = months[leading_idx]
    
    # Calculate target months for the behind actor
    target_months = leading_months - months_behind
    target_months = max(0, target_months)  # Can't go below 0
    
    # Interpolate to find the corresponding multiplier
    if target_months <= months[0]:
        return multipliers[0]
    elif target_months >= months[-1]:
        return multipliers[-1]
    else:
        # Linear interpolation between points
        for i in range(len(months)-1):
            if months[i] <= target_months <= months[i+1]:
                ratio = (target_months - months[i]) / (months[i+1] - months[i])
                return multipliers[i] + ratio * (multipliers[i+1] - multipliers[i])
    
    return multipliers[0]  # Fallback

# Other nation shares (more dynamic)
OTHER_NATION_SHARES = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]

# Coalition sabotage and neutralization parameters (Phase 2 transition parameters)
# These represent coalition capabilities to disrupt non-coalition actors
COALITION_DISRUPTION = {
    # Sabotage slowdown factor distributions (multiplier on actor's progress, <1.0 = slowed down)
    'sabotage_slowdown_factor': {
        'black_sites': DistributionParams(0.3, 0.6, 0.9),  # Moderate to high sabotage
        'criminal_orgs': DistributionParams(0.1, 0.4, 0.8),  # High sabotage potential
        'nation_states': DistributionParams(0.6, 0.8, 0.95),  # Limited sabotage (diplomatic)
        'other_nations': DistributionParams(0.4, 0.7, 0.9),  # Variable sabotage
    },
    
    # Neutralization probability per time period (chance actor is completely shut down)
    'neutralization_probability': {
        'black_sites': DistributionParams(0.02, 0.05, 0.15),  # 2-15% chance per period
        'criminal_orgs': DistributionParams(0.05, 0.12, 0.25),  # 5-25% chance per period
        'nation_states': DistributionParams(0.001, 0.01, 0.03),  # Very low (diplomatic consequences)
        'other_nations': DistributionParams(0.01, 0.03, 0.08),  # Low to moderate
    },
    
    # Coalition detection capability (affects sabotage success)
    'detection_capability': {
        'coalition_intelligence_level': DistributionParams(3.5, 4.2, 4.8),  # High detection (1-5 scale)
        'coalition_cyber_capability': DistributionParams(4.0, 4.5, 5.0),  # Very high cyber (1-5 scale)
    }
}

# Threat levels mapping (continuous -> discrete)
THREAT_LEVELS = ['OC2', 'OC3', 'OC4', 'OC5', 'OC6']

def continuous_to_threat_level(continuous_value: float) -> str:
    """Convert continuous threat value (1-5) to discrete threat level"""
    if continuous_value <= 2.5:
        return 'OC2'
    elif continuous_value <= 3.5:
        return 'OC3'  
    elif continuous_value <= 4.5:
        return 'OC4'
    elif continuous_value <= 5.5:
        return 'OC5'
    else:
        return 'OC6'

# Coalition member share distributions (controls US/China/EU allocation)
COALITION_INTERNAL_SHARES = {
    'us_base_share': DistributionParams(0.4, 0.6, 0.75),  # US gets 40-75% of coalition hardware
    'china_base_share': DistributionParams(0.1, 0.25, 0.4),  # China gets 10-40% of coalition hardware  
    'eu_base_share': DistributionParams(0.1, 0.15, 0.25),  # EU gets 10-25% of coalition hardware
    # Rest of coalition gets remainder
}

# Random seed configuration
RANDOM_SEED_CONFIG = {
    'default_seed': 42,
    'seed_description': 'Change random_seed parameter in ActorGenerator.__init__(random_seed=X) to modify randomization'
}

# Time periods for simulation
TIME_PERIODS = {
    'quarterly_end_year': 2035,
    'annual_end_year': 2042,
    'quarter_length_months': 4
}