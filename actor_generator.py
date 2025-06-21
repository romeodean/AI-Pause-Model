"""
AI Pause Model - Actor Generation (Phase 1)
Updated with Dirichlet distributions for proper share sampling
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random
from parameters import *

@dataclass
class Actor:
    """Represents an AI actor in the simulation"""
    name: str
    actor_type: str
    
    # Hardware characteristics
    compute_h100e: float = 0.0
    memory_tb: float = 0.0
    bandwidth_tb_s: float = 0.0
    num_clusters: int = 0
    cluster_sizes: List[float] = field(default_factory=list)
    
    # Hardware lifetime (for Phase 2 decay modeling)
    compute_lifetime_years: float = 3.0
    memory_lifetime_years: float = 3.0
    bandwidth_lifetime_years: float = 3.0
    
    # Software/R&D characteristics  
    purchasing_power_billion_usd: float = 0.0
    ai_rd_prog_multiplier: float = 1.0
    months_behind_coalition: float = 0.0
    base_ai_research_talent: float = 0.0
    
    # Security and defense
    threat_level: str = 'OC2'
    security_level: int = 1
    military_defense_level: int = 1
    
    # Phase 2 transition parameters
    sabotage_slowdown_factor: float = 1.0  # How much coalition slows this actor
    neutralization_probability: float = 0.0  # Chance of being shut down per period
    
    # Collaboration
    collaboration_partners: List[str] = field(default_factory=list)
    collaboration_efficiency: Dict[str, float] = field(default_factory=dict)
    
    # Status flags
    is_coalition_member: bool = False
    is_neutralized: bool = False

class ActorGenerator:
    """Generates actors for the AI pause simulation"""
    def __init__(self, random_seed: int = 42):
        """Initialize the generator with a random seed for reproducibility"""
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Sample global parameters first
        self.global_compute = self._sample_from_distribution(HARDWARE_GLOBAL['compute_million_h100e']) * 1e6
        self.global_memory = self._sample_from_distribution(HARDWARE_GLOBAL['memory_million_tb']) * 1e6
        self.global_bandwidth = self._sample_from_distribution(HARDWARE_GLOBAL['bandwidth_million_tb_s']) * 1e6
        
        # Sample hardware shares using Dirichlet (ensures they sum to 1)
        self.hardware_shares = sample_hardware_shares()
        
        # Sample Dirichlet distributions for internal allocations
        self.coalition_shares = sample_coalition_internal_shares()
        self.criminal_shares = sample_criminal_org_shares()
        self.nation_shares = sample_nation_state_shares()
        self.black_site_shares = sample_black_site_shares()
        
        # Sample open source months behind once for all actors that use it
        self.open_source_months_behind = self._sample_from_distribution(REFERENCE_MONTHS_BEHIND['open_source'])
        
        print(f"Global totals sampled:")
        print(f"  Compute: {self.global_compute/1e6:.1f}M H100e")
        print(f"  Memory: {self.global_memory/1e6:.1f}M TB")
        print(f"  Bandwidth: {self.global_bandwidth/1e6:.1f}M TB/s")
        print(f"Hardware shares: {self.hardware_shares}")
        print(f"Coalition internal shares: {self.coalition_shares}")
        print(f"Black site shares sampled: US({len(self.black_site_shares['us_sites'])} sites), China({len(self.black_site_shares['china_sites'])} sites), EU({len(self.black_site_shares['eu_sites'])} sites), Other({len(self.black_site_shares['other_coalition_sites'])} sites)")

    def _sample_from_distribution(self, dist_params: DistributionParams) -> float:
        """Sample from a lognormal distribution based on percentiles"""
        mu, sigma = dist_params.to_lognormal_params()
        return np.random.lognormal(mu, sigma)
    
    def _allocate_cluster_sizes(self, total_compute: float, is_coalition: bool) -> Tuple[int, List[float]]:
        """Allocate compute across clusters using Dirichlet distributions"""
        return sample_cluster_allocations(total_compute, is_coalition)
    
    def _get_months_behind(self, actor_type: str) -> float:
        """Get months behind coalition for actor type"""
        if actor_type in MONTHS_BEHIND:
            months_behind_value = MONTHS_BEHIND[actor_type]
            if months_behind_value == 'use_open_source':
                return self.open_source_months_behind
            else:
                return self._sample_from_distribution(months_behind_value)
        else:
            # Default to open source months behind
            return self.open_source_months_behind
    
    def _get_ai_research_talent(self, actor_type: str) -> float:
        """Get AI research talent for actor type"""
        if actor_type in AI_RESEARCH_TALENT:
            return self._sample_from_distribution(AI_RESEARCH_TALENT[actor_type])
        elif 'criminal' in actor_type.lower():
            return self._sample_from_distribution(AI_RESEARCH_TALENT['criminal_orgs'])
        elif 'black_site' in actor_type.lower():
            return self._sample_from_distribution(AI_RESEARCH_TALENT['random_other_black_sites'])
        else:
            return self._sample_from_distribution(AI_RESEARCH_TALENT['criminal_orgs'])
    
    def _get_purchasing_power(self, actor_type: str) -> float:
        """Get purchasing power for actor type"""
        if 'coalition' in actor_type.lower():
            return self._sample_from_distribution(PURCHASING_POWER['coalition'])
        elif 'criminal' in actor_type.lower():
            if 'major' in actor_type.lower():
                return self._sample_from_distribution(PURCHASING_POWER['criminal_orgs_major'])
            else:
                return self._sample_from_distribution(PURCHASING_POWER['criminal_orgs_minor'])
        elif actor_type == 'russia':
            return self._sample_from_distribution(PURCHASING_POWER['russia'])
        elif actor_type == 'iran':
            return self._sample_from_distribution(PURCHASING_POWER['iran'])
        elif 'black_site' in actor_type.lower():
            return self._sample_from_distribution(PURCHASING_POWER['black_sites'])
        else:
            return self._sample_from_distribution(PURCHASING_POWER['other_nations'])
    
    def _assign_threat_level(self, actor_type: str) -> str:
        """Assign threat level based on actor type using distributions"""
        if 'coalition' in actor_type.lower():
            threat_dist = THREAT_LEVEL_DISTRIBUTIONS['coalition']
        elif 'black_site' in actor_type.lower():
            threat_dist = THREAT_LEVEL_DISTRIBUTIONS['black_sites']
        elif 'criminal' in actor_type.lower():
            threat_dist = THREAT_LEVEL_DISTRIBUTIONS['criminal_orgs']
        elif actor_type in ['russia', 'iran', 'north_korea', 'iraq', 'syria']:
            threat_dist = THREAT_LEVEL_DISTRIBUTIONS['nation_states']
        else:
            threat_dist = THREAT_LEVEL_DISTRIBUTIONS['other_nations']
        
        continuous_value = self._sample_from_distribution(threat_dist)
        return continuous_to_threat_level(continuous_value)
    
    def _assign_security_level(self, actor_type: str) -> int:
        """Assign security level (1-5) based on actor type"""
        if 'coalition' in actor_type.lower():
            security_dist = SECURITY_LEVEL_DISTRIBUTIONS['coalition']
        elif 'black_site' in actor_type.lower():
            security_dist = SECURITY_LEVEL_DISTRIBUTIONS['black_sites']
        elif 'criminal' in actor_type.lower():
            security_dist = SECURITY_LEVEL_DISTRIBUTIONS['criminal_orgs']
        elif actor_type in ['russia', 'iran', 'north_korea', 'iraq', 'syria']:
            security_dist = SECURITY_LEVEL_DISTRIBUTIONS['nation_states']
        else:
            security_dist = SECURITY_LEVEL_DISTRIBUTIONS['other_nations']
        
        return max(1, min(5, int(round(self._sample_from_distribution(security_dist)))))
    
    def _assign_military_defense_level(self, actor_type: str) -> int:
        """Assign military defense level (1-5) based on actor type"""
        if 'coalition' in actor_type.lower():
            defense_dist = MILITARY_DEFENSE_DISTRIBUTIONS['coalition']
        elif 'black_site' in actor_type.lower():
            defense_dist = MILITARY_DEFENSE_DISTRIBUTIONS['black_sites']
        elif 'criminal' in actor_type.lower():
            defense_dist = MILITARY_DEFENSE_DISTRIBUTIONS['criminal_orgs']
        elif actor_type in ['russia', 'iran', 'north_korea', 'iraq', 'syria']:
            defense_dist = MILITARY_DEFENSE_DISTRIBUTIONS['nation_states']
        else:
            defense_dist = MILITARY_DEFENSE_DISTRIBUTIONS['other_nations']
        
        return max(1, min(5, int(round(self._sample_from_distribution(defense_dist)))))
    
    def _assign_coalition_disruption_params(self, actor_type: str) -> Tuple[float, float]:
        """Assign sabotage and neutralization parameters for non-coalition actors"""
        if 'coalition' in actor_type.lower():
            return 1.0, 0.0  # Coalition doesn't get sabotaged
        
        if 'black_site' in actor_type.lower():
            sabotage_dist = COALITION_DISRUPTION['sabotage_slowdown_factor']['black_sites']
            neutralization_dist = COALITION_DISRUPTION['neutralization_probability']['black_sites']
        elif 'criminal' in actor_type.lower():
            sabotage_dist = COALITION_DISRUPTION['sabotage_slowdown_factor']['criminal_orgs']
            neutralization_dist = COALITION_DISRUPTION['neutralization_probability']['criminal_orgs']
        elif actor_type in ['russia', 'iran', 'north_korea', 'iraq', 'syria']:
            sabotage_dist = COALITION_DISRUPTION['sabotage_slowdown_factor']['nation_states']
            neutralization_dist = COALITION_DISRUPTION['neutralization_probability']['nation_states']
        else:
            sabotage_dist = COALITION_DISRUPTION['sabotage_slowdown_factor']['other_nations']
            neutralization_dist = COALITION_DISRUPTION['neutralization_probability']['other_nations']
        
        sabotage_factor = self._sample_from_distribution(sabotage_dist)
        neutralization_prob = self._sample_from_distribution(neutralization_dist)
        
        return sabotage_factor, neutralization_prob
    
    def _create_actor(self, name: str, actor_type: str, compute_share: float, is_coalition: bool = False) -> Actor:
        """Create an actor with all properties"""
        # Hardware allocation
        actor_compute = self.global_compute * compute_share
        actor_memory = actor_compute * HARDWARE_SCALING['memory_per_h100e_tb']
        actor_bandwidth = actor_compute * HARDWARE_SCALING['bandwidth_per_h100e_tb_s']
        
        num_clusters, cluster_sizes = self._allocate_cluster_sizes(actor_compute, is_coalition)
        
        # Hardware lifetimes
        compute_lifetime = self._sample_from_distribution(HARDWARE_LIFETIME['compute_lifetime_years'])
        memory_lifetime = self._sample_from_distribution(HARDWARE_LIFETIME['memory_lifetime_years'])
        bandwidth_lifetime = self._sample_from_distribution(HARDWARE_LIFETIME['bandwidth_lifetime_years'])
        
        # AI capabilities
        months_behind = self._get_months_behind(actor_type)
        research_talent = self._get_ai_research_talent(actor_type)
        purchasing_power = self._get_purchasing_power(actor_type)
        
        # Security and defense
        threat_level = self._assign_threat_level(actor_type)
        security_level = self._assign_security_level(actor_type)
        military_defense_level = self._assign_military_defense_level(actor_type)
        
        # Coalition disruption parameters
        sabotage_factor, neutralization_prob = self._assign_coalition_disruption_params(actor_type)
        
        return Actor(
            name=name,
            actor_type=actor_type,
            compute_h100e=actor_compute,
            memory_tb=actor_memory,
            bandwidth_tb_s=actor_bandwidth,
            num_clusters=num_clusters,
            cluster_sizes=cluster_sizes,
            compute_lifetime_years=compute_lifetime,
            memory_lifetime_years=memory_lifetime,
            bandwidth_lifetime_years=bandwidth_lifetime,
            purchasing_power_billion_usd=purchasing_power,
            ai_rd_prog_multiplier=1.0,  # Will be updated later
            months_behind_coalition=months_behind,
            base_ai_research_talent=research_talent,
            threat_level=threat_level,
            security_level=security_level,
            military_defense_level=military_defense_level,
            sabotage_slowdown_factor=sabotage_factor,
            neutralization_probability=neutralization_prob,
            is_coalition_member=is_coalition
        )
    
    def generate_coalition_actors(self) -> List[Actor]:
        """Generate coalition member actors using Dirichlet-sampled shares"""
        actors = []
        
        # Get coalition's total share and internal allocations
        coalition_share = self.hardware_shares['coalition']
        
        # Coalition members with their Dirichlet-sampled shares
        coalition_members = [
            ('US', self.coalition_shares['us_share'], 'coalition'),
            ('China', self.coalition_shares['china_share'], 'coalition'),
            ('EU', self.coalition_shares['eu_share'], 'coalition'),
            ('Rest of Coalition', self.coalition_shares['rest_share'], 'coalition')
        ]
        
        for name, share, actor_type in coalition_members:
            if share <= 0:
                continue
            
            actor = self._create_actor(name, actor_type, coalition_share * share, is_coalition=True)
            
            # Coalition gets leading AI R&D multiplier
            actor.ai_rd_prog_multiplier = self._sample_from_distribution(SOFTWARE_PARAMS['leading_internal_ai_rd_mult'])
            actor.months_behind_coalition = 0.0  # Coalition is the reference
            
            actors.append(actor)
        
        return actors
    
    def generate_black_site_actors(self) -> List[Actor]:
        """Generate black site actors using Dirichlet-sampled shares"""
        actors = []
        
        # US Black Sites
        us_total_share = self.hardware_shares['us_secret_black_sites']
        for i, site_share in enumerate(self.black_site_shares['us_sites']):
            share = us_total_share * site_share
            name = f"US Black Site {i+1}"
            actor = self._create_actor(name, 'us_secret_black_sites', share, is_coalition=True)
            actors.append(actor)
        
        # China Black Sites  
        china_total_share = self.hardware_shares['china_secret_black_sites']
        for i, site_share in enumerate(self.black_site_shares['china_sites']):
            share = china_total_share * site_share
            name = f"China Black Site {i+1}"
            actor = self._create_actor(name, 'china_secret_black_sites', share, is_coalition=True)
            actors.append(actor)
        
        # EU Black Sites
        eu_total_share = self.hardware_shares['eu_secret_black_sites']
        for i, site_share in enumerate(self.black_site_shares['eu_sites']):
            share = eu_total_share * site_share
            name = f"EU Black Site {i+1}"
            actor = self._create_actor(name, 'eu_secret_black_sites', share, is_coalition=True)
            actors.append(actor)
        
        # Other Coalition Black Sites
        other_total_share = self.hardware_shares['random_other_black_sites']
        for i, site_share in enumerate(self.black_site_shares['other_coalition_sites']):
            share = other_total_share * site_share
            name = f"Coalition Black Site {i+1}"
            actor = self._create_actor(name, 'random_other_black_sites', share, is_coalition=True)
            actors.append(actor)
        
        return actors
    
    def generate_criminal_org_actors(self) -> List[Actor]:
        """Generate criminal organization actors using Dirichlet-sampled shares"""
        actors = []
        
        total_criminal_share = self.hardware_shares['all_criminal_orgs']
        
        # Get Dirichlet-sampled shares
        major_shares = self.criminal_shares['major_shares']
        minor_shares = self.criminal_shares['minor_shares']
        
        # Create major criminal orgs
        for i, relative_share in enumerate(major_shares):
            share = total_criminal_share * relative_share
            name = f"Criminal Org #{i+1}"
            actor = self._create_actor(name, 'criminal_org_major', share, is_coalition=False)
            actors.append(actor)
        
        # Create minor criminal orgs
        for i, relative_share in enumerate(minor_shares):
            share = total_criminal_share * relative_share
            name = f"Criminal Org #{i+len(major_shares)+1}"
            actor = self._create_actor(name, 'criminal_org_minor', share, is_coalition=False)
            actors.append(actor)
        
        return actors
    
    def generate_nation_state_actors(self) -> List[Actor]:
        """Generate nation state actors using Dirichlet-sampled shares"""
        actors = []
        
        # Get total share for all nation states (now unified)
        total_nation_share = self.hardware_shares['all_nation_states']
        
        # Specific nation states with Dirichlet-sampled shares
        specific_nations = ['russia', 'iran', 'north_korea', 'iraq', 'syria']
        
        for i, nation in enumerate(specific_nations):
            if nation == 'russia':
                share = total_nation_share * self.nation_shares['russia']
            elif nation == 'iran':
                share = total_nation_share * self.nation_shares['iran']
            elif nation == 'north_korea':
                share = total_nation_share * self.nation_shares['north_korea']
            elif nation == 'iraq':
                share = total_nation_share * self.nation_shares['iraq']
            elif nation == 'syria':
                share = total_nation_share * self.nation_shares['syria']
            
            name = nation.replace('_', ' ').title()
            actor = self._create_actor(name, nation, share, is_coalition=False)
            actors.append(actor)
        
        # Other non-coalition nations with Dirichlet-sampled shares
        other_nation_shares = self.nation_shares['other_nation_shares']
        
        for i, relative_share in enumerate(other_nation_shares):
            share = total_nation_share * relative_share
            name = f"Other Nation {i+1}"
            actor = self._create_actor(name, 'other_nation', share, is_coalition=False)
            actors.append(actor)
        
        return actors
    
    def generate_all_actors(self) -> List[Actor]:
        """Generate all actors for the simulation"""
        all_actors = []
        
        print("Generating coalition actors...")
        all_actors.extend(self.generate_coalition_actors())
        
        print("Generating black site actors...")
        all_actors.extend(self.generate_black_site_actors())
        
        print("Generating criminal organization actors...")
        all_actors.extend(self.generate_criminal_org_actors())
        
        print("Generating nation state actors...")
        all_actors.extend(self.generate_nation_state_actors())
        
        # Post-process: Update AI R&D multipliers based on months behind
        self._update_rd_multipliers(all_actors)
        
        return all_actors
    
    def _update_rd_multipliers(self, actors: List[Actor]):
        """Update AI R&D multipliers based on months behind and other factors"""
        # Find the leading coalition multiplier for reference
        coalition_actors = [a for a in actors if a.is_coalition_member]
        if coalition_actors:
            leading_multiplier = max(a.ai_rd_prog_multiplier for a in coalition_actors)
        else:
            leading_multiplier = 4.0  # Default fallback
        
        for actor in actors:
            if actor.is_coalition_member:
                # Coalition actors keep their sampled multipliers
                continue
            
            # Convert months behind to appropriate multiplier
            actor.ai_rd_prog_multiplier = months_behind_to_multiplier(
                leading_multiplier, actor.months_behind_coalition
            )
            
            # Apply talent and hardware modifiers
            talent_multiplier = actor.base_ai_research_talent
            hardware_boost = 1.0 + np.log10(max(1, actor.compute_h100e / 1000)) * 0.05
            
            # Final multiplier
            actor.ai_rd_prog_multiplier *= talent_multiplier * hardware_boost
    
    def print_actor_statistics(self, actors: List[Actor]):
        """Print comprehensive statistics about generated actors"""
        print(f"\n{'='*60}")
        print(f"ACTOR GENERATION SUMMARY ({len(actors)} total actors)")
        print(f"{'='*60}")
        
        # Summary by type
        actor_types = {}
        for actor in actors:
            if actor.actor_type not in actor_types:
                actor_types[actor.actor_type] = []
            actor_types[actor.actor_type].append(actor)
        
        print(f"\nActors by type:")
        for actor_type, type_actors in actor_types.items():
            print(f"  {actor_type}: {len(type_actors)} actors")
        
        # Hardware totals
        total_compute = sum(actor.compute_h100e for actor in actors)
        total_memory = sum(actor.memory_tb for actor in actors)
        total_bandwidth = sum(actor.bandwidth_tb_s for actor in actors)
        
        print(f"\nHardware totals (should match global):")
        print(f"  Total compute: {total_compute/1e6:.1f}M H100e (target: {self.global_compute/1e6:.1f}M)")
        print(f"  Total memory: {total_memory/1e6:.1f}M TB (target: {self.global_memory/1e6:.1f}M)")
        print(f"  Total bandwidth: {total_bandwidth/1e6:.1f}M TB/s (target: {self.global_bandwidth/1e6:.1f}M)")
        
        # Coalition vs non-coalition
        coalition_actors = [a for a in actors if a.is_coalition_member]
        non_coalition_actors = [a for a in actors if not a.is_coalition_member]
        
        coalition_compute = sum(a.compute_h100e for a in coalition_actors)
        non_coalition_compute = sum(a.compute_h100e for a in non_coalition_actors)
        
        print(f"\nCoalition vs Non-Coalition:")
        print(f"  Coalition: {len(coalition_actors)} actors, {coalition_compute/total_compute:.1%} of compute")
        print(f"  Non-Coalition: {len(non_coalition_actors)} actors, {non_coalition_compute/total_compute:.1%} of compute")
        
        # Threat level distribution
        threat_counts = {}
        for actor in actors:
            threat_counts[actor.threat_level] = threat_counts.get(actor.threat_level, 0) + 1
        
        print(f"\nThreat level distribution:")
        for level in THREAT_LEVELS:
            count = threat_counts.get(level, 0)
            print(f"  {level}: {count} actors")
        
        # Top actors by compute
        top_actors = sorted(actors, key=lambda a: a.compute_h100e, reverse=True)[:10]
        print(f"\nTop 10 actors by compute:")
        for i, actor in enumerate(top_actors, 1):
            print(f"  {i:2d}. {actor.name:25s} {actor.compute_h100e/1e6:8.2f}M H100e ({actor.actor_type})")
        
        # Research capability distribution
        print(f"\nAI Research Capability Summary:")
        talent_values = [a.base_ai_research_talent for a in actors]
        rd_mult_values = [a.ai_rd_prog_multiplier for a in actors]
        months_behind_values = [a.months_behind_coalition for a in actors if a.months_behind_coalition > 0]
        
        print(f"  AI Research Talent - Mean: {np.mean(talent_values):.2f}, Median: {np.median(talent_values):.2f}")
        print(f"  AI R&D Multiplier - Mean: {np.mean(rd_mult_values):.2f}, Median: {np.median(rd_mult_values):.2f}")
        if months_behind_values:
            print(f"  Months Behind Coalition - Mean: {np.mean(months_behind_values):.1f}, Median: {np.median(months_behind_values):.1f}")
        
        # Purchasing power
        purchasing_power_values = [a.purchasing_power_billion_usd for a in actors]
        print(f"  Purchasing Power - Mean: ${np.mean(purchasing_power_values):.1f}B, Median: ${np.median(purchasing_power_values):.1f}B")
        
        # Cluster statistics
        all_clusters = []
        for actor in actors:
            all_clusters.extend(actor.cluster_sizes)
        
        large_clusters = [c for c in all_clusters if c > 100000]
        medium_clusters = [c for c in all_clusters if 1000 <= c <= 100000]
        small_clusters = [c for c in all_clusters if c < 1000]
        
        print(f"\nCluster size distribution:")
        print(f"  Large clusters (>100K H100e): {len(large_clusters)} ({sum(large_clusters)/total_compute:.1%} of compute)")
        print(f"  Medium clusters (1K-100K H100e): {len(medium_clusters)} ({sum(medium_clusters)/total_compute:.1%} of compute)")
        print(f"  Small clusters (<1K H100e): {len(small_clusters)} ({sum(small_clusters)/total_compute:.1%} of compute)")

        # Dirichlet sampling validation
        print(f"\nDirichlet Sampling Validation:")
        print(f"  Coalition shares sum: {sum(self.coalition_shares.values()):.6f} (should be 1.0)")
        print(f"  Criminal major shares sum: {sum(self.criminal_shares['major_shares']):.6f}")
        print(f"  Criminal minor shares sum: {sum(self.criminal_shares['minor_shares']):.6f}")
        print(f"  Nation state shares sum: {sum([self.nation_shares[k] for k in ['russia', 'iran', 'north_korea', 'iraq', 'syria']] + list(self.nation_shares['other_nation_shares'])):.6f} (should be 1.0)")
        print(f"  US black sites sum: {sum(self.black_site_shares['us_sites']):.6f} (should be 1.0)")
        print(f"  China black sites sum: {sum(self.black_site_shares['china_sites']):.6f} (should be 1.0)")
        print(f"  EU black sites sum: {sum(self.black_site_shares['eu_sites']):.6f} (should be 1.0)")
        print(f"  Other coalition black sites sum: {sum(self.black_site_shares['other_coalition_sites']):.6f} (should be 1.0)")

def save_actors_to_csv(actors: List[Actor], filename: str = "ai_pause_actors.csv"):
    """Save actors to CSV file for further analysis"""
    data = []
    for actor in actors:
        data.append({
            'name': actor.name,
            'actor_type': actor.actor_type,
            'compute_h100e': actor.compute_h100e,
            'memory_tb': actor.memory_tb,
            'bandwidth_tb_s': actor.bandwidth_tb_s,
            'num_clusters': actor.num_clusters,
            'largest_cluster_size': max(actor.cluster_sizes) if actor.cluster_sizes else 0,
            'compute_lifetime_years': actor.compute_lifetime_years,
            'memory_lifetime_years': actor.memory_lifetime_years,
            'bandwidth_lifetime_years': actor.bandwidth_lifetime_years,
            'purchasing_power_billion_usd': actor.purchasing_power_billion_usd,
            'ai_rd_prog_multiplier': actor.ai_rd_prog_multiplier,
            'months_behind_coalition': actor.months_behind_coalition,
            'base_ai_research_talent': actor.base_ai_research_talent,
            'threat_level': actor.threat_level,
            'security_level': actor.security_level,
            'military_defense_level': actor.military_defense_level,
            'sabotage_slowdown_factor': actor.sabotage_slowdown_factor,
            'neutralization_probability': actor.neutralization_probability,
            'is_coalition_member': actor.is_coalition_member,
            'is_neutralized': actor.is_neutralized
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\nActors saved to {filename}")
    return df

def main():
    """Main function to generate and analyze actors"""
    print("Starting AI Pause Model - Phase 1: Actor Generation")
    print("=" * 60)
    
    # Generate actors
    generator = ActorGenerator(random_seed=42)
    actors = generator.generate_all_actors()
    
    # Print statistics
    generator.print_actor_statistics(actors)
    
    # Save to CSV
    df = save_actors_to_csv(actors)
    
    print(f"\n{'='*60}")
    print("Phase 1 Complete: All actors generated successfully!")
    print(f"Generated {len(actors)} actors with hardware and software characteristics")
    print("Ready for Phase 2: Time-step simulation")
    
    return actors, df

if __name__ == "__main__":
    actors, df = main()