"""
AI Pause Model - Parameters Configuration
Updated with Dirichlet distributions for proper share sampling
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
        # For lognormal: ln(p85/p50) ≈ 1.036*sigma, ln(p50/p15) ≈ 1.036*sigma
        upper_ratio = np.log(self.p85 / self.p50) / 1.036
        lower_ratio = np.log(self.p50 / self.p15) / 1.036
        sigma = (upper_ratio + lower_ratio) / 2
        mu = np.log(self.p50)
        return mu, sigma

# Global Hardware Parameters (March 2027 starting conditions)
HARDWARE_GLOBAL = {
    'compute_million_h100e': DistributionParams(20, 50, 100),
    'memory_million_tb': DistributionParams(1.4, 3.5, 7.0),
    'bandwidth_million_tb_s': DistributionParams(28, 70, 140),
}

# Hardware shares by actor type - Updated for cleaner nation state allocation
HARDWARE_SHARES_DIRICHLET = {
    'coalition': 85,                         # 0.9 * 100
    'all_criminal_orgs': 1,                  # 0.01 * 100  
    'us_secret_black_sites': 2,              # 0.02 * 100
    'china_secret_black_sites': 2,           # 0.02 * 100
    'eu_secret_black_sites': 1,              # 0.01 * 100
    'random_other_black_sites': 1,         # 0.009 * 100
    'all_nation_states': 8,               # Combined: russia + iran + other nations
}

# Black site internal shares using Dirichlet
BLACK_SITE_DIRICHLET = {
    'us_black_sites': {
        'site_alphas': [1, 1, 1]  # 3 sites, equal probability
    },
    'china_black_sites': {
        'site_alphas': [1, 1, 1]  # 3 sites, equal probability  
    },
    'eu_black_sites': {
        'site_alphas': [1] * 10   # 10 sites, equal probability
    },
    'other_coalition_black_sites': {
        'site_alphas': [1] * 20   # 20 sites, equal probability
    }
}

def sample_black_site_shares():
    """Sample black site shares within each country/region using Dirichlet"""
    return {
        'us_sites': np.random.dirichlet(BLACK_SITE_DIRICHLET['us_black_sites']['site_alphas']),
        'china_sites': np.random.dirichlet(BLACK_SITE_DIRICHLET['china_black_sites']['site_alphas']),
        'eu_sites': np.random.dirichlet(BLACK_SITE_DIRICHLET['eu_black_sites']['site_alphas']),
        'other_coalition_sites': np.random.dirichlet(BLACK_SITE_DIRICHLET['other_coalition_black_sites']['site_alphas'])
    }

def sample_hardware_shares():
    """Sample hardware shares that sum to 1"""
    alphas = list(HARDWARE_SHARES_DIRICHLET.values())
    names = list(HARDWARE_SHARES_DIRICHLET.keys())
    shares = np.random.dirichlet(alphas)
    return dict(zip(names, shares))

# Coalition member share distributions using Dirichlet
COALITION_INTERNAL_DIRICHLET = {
    'us_alpha': 75,      # Reflects 40-75% range, higher concentration
    'china_alpha': 15,   # Reflects 10-40% range  
    'eu_alpha': 5,      # Reflects 10-25% range
    'rest_alpha': 5      # Remainder gets smaller share
}

def sample_coalition_internal_shares():
    """Sample coalition member shares using Dirichlet distribution"""
    alphas = [
        COALITION_INTERNAL_DIRICHLET['us_alpha'],
        COALITION_INTERNAL_DIRICHLET['china_alpha'], 
        COALITION_INTERNAL_DIRICHLET['eu_alpha'],
        COALITION_INTERNAL_DIRICHLET['rest_alpha']
    ]
    shares = np.random.dirichlet(alphas)
    return {
        'us_share': shares[0],
        'china_share': shares[1],
        'eu_share': shares[2], 
        'rest_share': shares[3]
    }

# Criminal organization hierarchy using Dirichlet
CRIMINAL_ORG_DIRICHLET = {
    # Major orgs get higher alphas (more concentrated, power law-like)
    'major_alphas': [20, 15, 12, 10, 8, 6, 5, 4, 3, 2],  # Decreasing concentration
    'minor_alphas': [1] * 25,  # More uniform for minor orgs
    'major_total_weight': 75,  # Major orgs get 75% of total criminal compute
    'minor_total_weight': 25   # Minor orgs get 25% of total criminal compute
}

def sample_criminal_org_shares():
    """Sample criminal organization shares using Dirichlet distributions"""
    # Sample major org shares
    major_shares = np.random.dirichlet(CRIMINAL_ORG_DIRICHLET['major_alphas'])
    
    # Sample minor org shares  
    minor_shares = np.random.dirichlet(CRIMINAL_ORG_DIRICHLET['minor_alphas'])
    
    # Weight by major/minor allocation
    major_weight = CRIMINAL_ORG_DIRICHLET['major_total_weight'] / 100
    minor_weight = CRIMINAL_ORG_DIRICHLET['minor_total_weight'] / 100
    
    return {
        'major_shares': major_shares * major_weight,
        'minor_shares': minor_shares * minor_weight
    }

# Nation state shares using Dirichlet
NATION_STATE_DIRICHLET = {
    # Specific nation alphas (higher values = more likely to get larger shares)
    'russia_alpha': 15,
    'iran_alpha': 2,
    'north_korea_alpha': 1,
    'iraq_alpha': 1,
    'syria_alpha': 1,
    
    # Other nations alphas (decreasing concentration)
    'other_nations_alphas': [3, 2.5, 2, 1.5, 1.2, 1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2]
}

def sample_nation_state_shares():
    """Sample nation state shares using Dirichlet distribution"""
    # Sample specific nation shares
    specific_alphas = [
        NATION_STATE_DIRICHLET['russia_alpha'],
        NATION_STATE_DIRICHLET['iran_alpha'],
        NATION_STATE_DIRICHLET['north_korea_alpha'],
        NATION_STATE_DIRICHLET['iraq_alpha'],
        NATION_STATE_DIRICHLET['syria_alpha']
    ]
    
    # Sample other nations shares
    other_alphas = NATION_STATE_DIRICHLET['other_nations_alphas']
    
    # Combine all alphas for joint sampling
    all_alphas = specific_alphas + other_alphas
    all_shares = np.random.dirichlet(all_alphas)
    
    # Split back into specific and other nations
    specific_shares = all_shares[:5]
    other_shares = all_shares[5:]
    
    return {
        'russia': specific_shares[0],
        'iran': specific_shares[1], 
        'north_korea': specific_shares[2],
        'iraq': specific_shares[3],
        'syria': specific_shares[4],
        'other_nation_shares': other_shares
    }

# Pre-deal hardware ownership
PRE_DEAL_OWNERSHIP = {
    'us_share': DistributionParams(0.5, 0.75, 0.85),
    'china_share': DistributionParams(0.05, 0.12, 0.25),
    'rest_of_world_share': DistributionParams(0.05, 0.13, 0.25),
    'non_agi_usage_share': DistributionParams(0.05, 0.1, 0.2),
}

# Cluster size distributions using Dirichlet
CLUSTER_SIZE_DIRICHLET = {
    'coalition': {
        # More concentrated - fewer, larger clusters
        'large_cluster_alphas': [50, 40, 30, 20, 20, 20, 15, 10, 10],  # 5 large clusters (>100K H100e)
        'medium_cluster_alphas': [8, 7, 6, 4, 3, 2, 1],   # 6 medium clusters (1K-100K H100e)  
        'small_cluster_alphas': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],             # 3 small clusters (<1K H100e)
        'large_cluster_share': 0.72,
        'medium_cluster_share': 0.23,
        'small_cluster_share': 0.05,
        'large_cluster_size_range': (100000, 10000000),
        'medium_cluster_size_range': (1000, 100000),
        'small_cluster_size_range': (10, 1000)
    },
    'non_coalition': {
        # Less concentrated - more, smaller clusters
        'large_cluster_alphas': [5, 3],                 # 2 large clusters  
        'medium_cluster_alphas': [3, 2, 2, 1, 1, 1, 1], # 7 medium clusters
        'small_cluster_alphas': [1] * 8,                # 8 small clusters
        'large_cluster_share': 0.12,
        'medium_cluster_share': 0.53,
        'small_cluster_share': 0.35,
        'large_cluster_size_range': (100000, 300000),
        'medium_cluster_size_range': (1000, 50000),
        'small_cluster_size_range': (10, 800)
    }
}

def sample_cluster_allocations(total_compute: float, is_coalition: bool):
    """Sample cluster sizes using Dirichlet distributions"""
    if is_coalition:
        config = CLUSTER_SIZE_DIRICHLET['coalition']
    else:
        config = CLUSTER_SIZE_DIRICHLET['non_coalition']
    
    clusters = []
    
    # Large clusters
    large_compute = total_compute * config['large_cluster_share']
    if large_compute > 100000:  # Only create if sufficient compute
        large_shares = np.random.dirichlet(config['large_cluster_alphas'])
        for share in large_shares:
            cluster_compute = large_compute * share
            clusters.append(cluster_compute)
    
    # Medium clusters  
    medium_compute = total_compute * config['medium_cluster_share']
    if medium_compute > 1000:
        medium_shares = np.random.dirichlet(config['medium_cluster_alphas'])
        for share in medium_shares:
            cluster_compute = medium_compute * share
            clusters.append(cluster_compute)
    
    # Small clusters
    small_compute = total_compute * config['small_cluster_share']
    if small_compute > 100:
        small_shares = np.random.dirichlet(config['small_cluster_alphas'])
        for share in small_shares:
            cluster_compute = small_compute * share
            clusters.append(cluster_compute)
    
    # Handle remainder as a single cluster if significant
    allocated_compute = sum(clusters)
    remainder = total_compute - allocated_compute
    if remainder > 50:  # Add remainder as final cluster
        clusters.append(remainder)
    
    return len(clusters), clusters

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