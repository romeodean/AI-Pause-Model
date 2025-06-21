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
    'pre_deal': {
        'large_cluster_share': DistributionParams(0.5, 0.66, 0.9),  # >100K H100e
        'medium_cluster_share': DistributionParams(0.09, 0.26, 0.4),  # 1K-100K H100e
        'small_cluster_share': DistributionParams(0.01, 0.08, 0.15),  # <1K H100e
    },
    'coalition': {
        'large_cluster_share': 0.72,  # Fixed value
        'medium_cluster_share': 0.23,
        'small_cluster_share': 0.05,  # Inferred
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

# Hardware decay parameters (from Random parameters sheet)
HARDWARE_DECAY = {
    'hbm_capacity_per_h100e_gb': {
        2023: 120, 2024: 100, 2025: 90, 2026: 80, 2027: 70
    },
    'hbm_bandwidth_per_h100e_tb_s': {
        2023: 2.4, 2024: 2.0, 2025: 1.8, 2026: 1.6, 2027: 1.4
    },
    'decay_rate': 0.8739351325  # Annual decay rate
}

# Actor definitions and counts
ACTOR_DEFINITIONS = {
    'coalition_members': ['us', 'china', 'eu', 'rest_of_coalition'],
    'rogue_black_sites': {
        'us_rogue': 3,
        'china_rogue': 3, 
        'eu_rogue': 10,
        'other_coalition_rogue': 50  # Mix of sanctioned and unsanctioned
    },
    'criminal_orgs': {
        'major_criminal_orgs': 5,  # #1 through #5
        'minor_criminal_orgs': 5
    },
    'non_coalition_nations': [
        'russia', 'north_korea', 'iran', 'iraq', 'syria'
    ],
    'other_nations': 10  # Additional non-coalition nations
}

# Collaboration likelihood tuples (actor1, actor2, efficiency_multiplier)
COLLABORATION_MATRIX = {
    ('iran', 'iraq'): 0.8,
    ('russia', 'syria'): 0.8,
    ('china', 'us_black_sites'): 0.0,
    ('north_korea', 'iran'): 0.6,
    ('russia', 'iran'): 0.7,
    # Criminal orgs have penalties when working together
    ('criminal_org', 'criminal_org'): 0.5,  # Generic criminal collaboration
}

# Threat levels (OC2 to OC5)
THREAT_LEVELS = ['OC2', 'OC3', 'OC4', 'OC5']

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

# Share distributions for subdividing actor groups
CRIMINAL_ORG_SHARES = [0.25, 0.20, 0.18, 0.15, 0.12, 0.08, 0.05, 0.04, 0.02, 0.01]  # Top 10 criminal orgs
OTHER_NATION_SHARES = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]  # For flexibility if more nations needed

# Time periods for simulation
TIME_PERIODS = {
    'quarterly_end_year': 2035,
    'annual_end_year': 2042,
    'quarter_length_months': 4
}