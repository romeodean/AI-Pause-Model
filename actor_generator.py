"""
AI Pause Model - Actor Generation (Phase 1)
Generates actors with hardware and software characteristics based on model parameters
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
    
    # Software/R&D characteristics  
    purchasing_power_billion_usd: float = 0.0
    ai_rd_prog_multiplier: float = 1.0
    months_behind_coalition: float = 0.0
    base_ai_research_talent: float = 0.0
    threat_level: str = 'OC2'
    
    # Collaboration
    collaboration_partners: List[str] = field(default_factory=list)
    collaboration_efficiency: Dict[str, float] = field(default_factory=dict)
    
    # Flags
    is_sanctioned: bool = False
    is_coalition_member: bool = False

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
        
        # Sample open source months behind once for all actors that use it
        self.open_source_months_behind = self._sample_from_distribution(REFERENCE_MONTHS_BEHIND['open_source'])
        
        print(f"Global totals sampled:")
        print(f"  Compute: {self.global_compute/1e6:.1f}M H100e")
        print(f"  Memory: {self.global_memory/1e6:.1f}M TB")
        print(f"  Bandwidth: {self.global_bandwidth/1e6:.1f}M TB/s")
        print(f"  Open source months behind: {self.open_source_months_behind:.1f} months")
    
    def _sample_from_distribution(self, dist_params: DistributionParams) -> float:
        """Sample from a lognormal distribution based on percentiles"""
        mu, sigma = dist_params.to_lognormal_params()
        return np.random.lognormal(mu, sigma)
    
    def _allocate_cluster_sizes(self, total_compute: float, actor_type: str) -> Tuple[int, List[float]]:
        """Allocate compute across clusters based on size distributions"""
        # Determine if coalition or non-coalition
        is_coalition = actor_type in ['coalition', 'us', 'china', 'eu'] or 'coalition' in actor_type.lower()
        
        if is_coalition:
            # Use fixed coalition cluster distribution
            large_share = CLUSTER_SIZES['coalition']['large_cluster_share']
            medium_share = CLUSTER_SIZES['coalition']['medium_cluster_share']
            small_share = CLUSTER_SIZES['coalition']['small_cluster_share']
        else:
            # Use pre-deal distributions for non-coalition actors
            large_share = self._sample_from_distribution(CLUSTER_SIZES['pre_deal']['large_cluster_share'])
            medium_share = self._sample_from_distribution(CLUSTER_SIZES['pre_deal']['medium_cluster_share'])
            small_share = self._sample_from_distribution(CLUSTER_SIZES['pre_deal']['small_cluster_share'])
        
        # Normalize shares
        total_share = large_share + medium_share + small_share
        large_share /= total_share
        medium_share /= total_share
        small_share /= total_share
        
        clusters = []
        remaining_compute = total_compute
        
        # Large clusters (>100K H100e)
        large_compute = total_compute * large_share
        while large_compute > 100000 and remaining_compute > 100000:
            cluster_size = np.random.uniform(100000, min(remaining_compute, 500000))
            clusters.append(cluster_size)
            remaining_compute -= cluster_size
            large_compute -= cluster_size
        
        # Medium clusters (1K-100K H100e)
        medium_compute = total_compute * medium_share
        while medium_compute > 1000 and remaining_compute > 1000:
            cluster_size = np.random.uniform(1000, min(remaining_compute, 100000))
            clusters.append(cluster_size)
            remaining_compute -= cluster_size
            medium_compute -= cluster_size
        
        # Small clusters (<1K H100e)
        while remaining_compute > 100:
            cluster_size = min(remaining_compute, np.random.uniform(10, 1000))
            clusters.append(cluster_size)
            remaining_compute -= cluster_size
        
        return len(clusters), clusters
    
    def _get_months_behind(self, actor_type: str) -> float:
        """Get months behind coalition for actor type"""
        if actor_type in MONTHS_BEHIND:
            months_behind_value = MONTHS_BEHIND[actor_type]
            if months_behind_value == 'use_open_source':
                return self.open_source_months_behind
            else:
                return self._sample_from_distribution(months_behind_value)
        elif 'criminal' in actor_type.lower():
            # Use open source months behind
            return self.open_source_months_behind
        elif 'black_site' in actor_type.lower():
            return self.open_source_months_behind
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
            return self._sample_from_distribution(AI_RESEARCH_TALENT['criminal_orgs'])  # Conservative default
    
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
        """Assign threat level based on actor type"""
        if 'coalition' in actor_type.lower():
            return 'OC2'  # Lowest threat
        elif 'criminal' in actor_type.lower():
            return np.random.choice(['OC3', 'OC4', 'OC5'], p=[0.3, 0.5, 0.2])
        elif actor_type in ['russia', 'iran', 'north_korea']:
            return np.random.choice(['OC4', 'OC5'], p=[0.3, 0.7])
        elif 'black_site' in actor_type.lower():
            return np.random.choice(['OC3', 'OC4', 'OC5'], p=[0.2, 0.5, 0.3])
        else:
            return np.random.choice(['OC2', 'OC3', 'OC4'], p=[0.3, 0.5, 0.2])
    
    def generate_coalition_actors(self) -> List[Actor]:
        """Generate coalition member actors"""
        actors = []
        
        # Sample coalition's total share
        coalition_share = self._sample_from_distribution(HARDWARE_SHARES['coalition'])
        
        # Allocate within coalition based on pre-deal ownership
        us_share = self._sample_from_distribution(PRE_DEAL_OWNERSHIP['us_share'])
        china_share = self._sample_from_distribution(PRE_DEAL_OWNERSHIP['china_share'])
        other_share = 1 - us_share - china_share
        
        # Coalition members
        coalition_members = [
            ('US', us_share * 0.7),  # 70% of US share
            ('China', china_share * 0.8),  # 80% of China share  
            ('EU', other_share * 0.6),  # 60% of other share
            ('Rest of Coalition', us_share * 0.3 + china_share * 0.2 + other_share * 0.4)
        ]
        
        for name, share in coalition_members:
            actor_compute = self.global_compute * coalition_share * share
            actor_memory = self.global_memory * coalition_share * share
            actor_bandwidth = self.global_bandwidth * coalition_share * share
            
            num_clusters, cluster_sizes = self._allocate_cluster_sizes(actor_compute, name.lower())
            
            actor = Actor(
                name=name,
                actor_type='coalition',
                compute_h100e=actor_compute,
                memory_tb=actor_memory,
                bandwidth_tb_s=actor_bandwidth,
                num_clusters=num_clusters,
                cluster_sizes=cluster_sizes,
                purchasing_power_billion_usd=self._get_purchasing_power('coalition'),
                ai_rd_prog_multiplier=self._sample_from_distribution(SOFTWARE_PARAMS['leading_internal_ai_rd_mult']),
                months_behind_coalition=0.0,  # Coalition is the reference
                base_ai_research_talent=self._get_ai_research_talent('coalition'),
                threat_level='OC2',
                is_coalition_member=True
            )
            actors.append(actor)
        
        return actors
    
    def generate_black_site_actors(self) -> List[Actor]:
        """Generate black site actors"""
        actors = []
        
        # US black sites
        us_total_share = self._sample_from_distribution(HARDWARE_SHARES['us_secret_black_sites'])
        for i in range(ACTOR_DEFINITIONS['rogue_black_sites']['us_rogue']):
            share = us_total_share / ACTOR_DEFINITIONS['rogue_black_sites']['us_rogue']
            actor = self._create_black_site_actor(f"US Black Site {i+1}", share, 'us_secret_black_sites')
            actor.is_sanctioned = True
            actors.append(actor)
        
        # China black sites
        china_total_share = self._sample_from_distribution(HARDWARE_SHARES['china_secret_black_sites'])
        for i in range(ACTOR_DEFINITIONS['rogue_black_sites']['china_rogue']):
            share = china_total_share / ACTOR_DEFINITIONS['rogue_black_sites']['china_rogue']
            actor = self._create_black_site_actor(f"China Black Site {i+1}", share, 'china_secret_black_sites')
            actor.is_sanctioned = True
            actors.append(actor)
        
        # EU black sites
        eu_total_share = self._sample_from_distribution(HARDWARE_SHARES['eu_secret_black_sites'])
        for i in range(ACTOR_DEFINITIONS['rogue_black_sites']['eu_rogue']):
            share = eu_total_share / ACTOR_DEFINITIONS['rogue_black_sites']['eu_rogue']
            actor = self._create_black_site_actor(f"EU Black Site {i+1}", share, 'eu_secret_black_sites')
            actor.is_sanctioned = True
            actors.append(actor)
        
        # Other coalition black sites
        other_total_share = self._sample_from_distribution(HARDWARE_SHARES['random_other_black_sites'])
        for i in range(ACTOR_DEFINITIONS['rogue_black_sites']['other_coalition_rogue']):
            share = other_total_share / ACTOR_DEFINITIONS['rogue_black_sites']['other_coalition_rogue']
            actor = self._create_black_site_actor(f"Coalition Black Site {i+1}", share, 'random_other_black_sites')
            # Mix of sanctioned and unsanctioned
            actor.is_sanctioned = np.random.choice([True, False], p=[0.6, 0.4])
            actors.append(actor)
        
        return actors
    
    def _create_black_site_actor(self, name: str, share: float, actor_type: str) -> Actor:
        """Create a black site actor"""
        actor_compute = self.global_compute * share
        actor_memory = self.global_memory * share
        actor_bandwidth = self.global_bandwidth * share
        
        num_clusters, cluster_sizes = self._allocate_cluster_sizes(actor_compute, actor_type)
        
        return Actor(
            name=name,
            actor_type=actor_type,
            compute_h100e=actor_compute,
            memory_tb=actor_memory,
            bandwidth_tb_s=actor_bandwidth,
            num_clusters=num_clusters,
            cluster_sizes=cluster_sizes,
            purchasing_power_billion_usd=self._get_purchasing_power('black_site'),
            ai_rd_prog_multiplier=1.0,  # Will be updated based on talent/hardware
            months_behind_coalition=self._get_months_behind(actor_type),
            base_ai_research_talent=self._get_ai_research_talent(actor_type),
            threat_level=self._assign_threat_level(actor_type)
        )
    
    def generate_criminal_org_actors(self) -> List[Actor]:
        """Generate criminal organization actors with realistic share distribution"""
        actors = []
        
        total_criminal_share = self._sample_from_distribution(HARDWARE_SHARES['all_criminal_orgs'])
        
        # Use predefined shares for 10 criminal orgs
        criminal_shares = CRIMINAL_ORG_SHARES[:10]  # Take first 10 shares
        
        for i, relative_share in enumerate(criminal_shares):
            share = total_criminal_share * relative_share
            org_size = 'major' if i < 5 else 'minor'  # First 5 are major
            actor = self._create_criminal_actor(f"Criminal Org #{i+1}", share, org_size)
            actors.append(actor)
        
        return actors
    
    def _create_criminal_actor(self, name: str, share: float, org_size: str) -> Actor:
        """Create a criminal organization actor"""
        actor_compute = self.global_compute * share
        actor_memory = self.global_memory * share
        actor_bandwidth = self.global_bandwidth * share
        
        num_clusters, cluster_sizes = self._allocate_cluster_sizes(actor_compute, 'criminal')
        
        return Actor(
            name=name,
            actor_type=f'criminal_org_{org_size}',
            compute_h100e=actor_compute,
            memory_tb=actor_memory,
            bandwidth_tb_s=actor_bandwidth,
            num_clusters=num_clusters,
            cluster_sizes=cluster_sizes,
            purchasing_power_billion_usd=self._get_purchasing_power(f'criminal_{org_size}'),
            ai_rd_prog_multiplier=1.0,
            months_behind_coalition=self._get_months_behind('criminal'),
            base_ai_research_talent=self._get_ai_research_talent('criminal'),
            threat_level=self._assign_threat_level('criminal')
        )
    
    def generate_nation_state_actors(self) -> List[Actor]:
        """Generate nation state actors"""
        actors = []
        
        # Specific nation states
        nations = ['russia', 'north_korea', 'iran', 'iraq', 'syria']
        
        for nation in nations:
            if nation in HARDWARE_SHARES:
                share = self._sample_from_distribution(HARDWARE_SHARES[nation])
            else:
                # Use other_non_coalition_nations as default
                share = self._sample_from_distribution(HARDWARE_SHARES['other_non_coalition_nations']) / 5
            
            actor = self._create_nation_actor(nation.replace('_', ' ').title(), share, nation)
            actors.append(actor)
        
        # Other non-coalition nations with realistic share distribution
        other_nations_total_share = self._sample_from_distribution(HARDWARE_SHARES['other_non_coalition_nations'])
        # Subtract shares already allocated to specific nations
        remaining_share = other_nations_total_share - sum([
            self._sample_from_distribution(HARDWARE_SHARES.get(nation, HARDWARE_SHARES['other_non_coalition_nations'])) / 5
            for nation in nations if nation not in HARDWARE_SHARES
        ])
        
        # Use predefined shares for other nations
        other_nation_shares = OTHER_NATION_SHARES[:ACTOR_DEFINITIONS['other_nations']]
        # Normalize the shares to sum to 1
        total_relative = sum(other_nation_shares)
        normalized_shares = [s/total_relative for s in other_nation_shares]
        
        for i, relative_share in enumerate(normalized_shares):
            share = remaining_share * relative_share
            actor = self._create_nation_actor(f"Other Nation {i+1}", share, 'other_nation')
            actors.append(actor)
        
        return actors
    
    def _create_nation_actor(self, name: str, share: float, nation_type: str) -> Actor:
        """Create a nation state actor"""
        actor_compute = self.global_compute * share
        actor_memory = self.global_memory * share
        actor_bandwidth = self.global_bandwidth * share
        
        num_clusters, cluster_sizes = self._allocate_cluster_sizes(actor_compute, nation_type)
        
        return Actor(
            name=name,
            actor_type=nation_type,
            compute_h100e=actor_compute,
            memory_tb=actor_memory,
            bandwidth_tb_s=actor_bandwidth,
            num_clusters=num_clusters,
            cluster_sizes=cluster_sizes,
            purchasing_power_billion_usd=self._get_purchasing_power(nation_type),
            ai_rd_prog_multiplier=1.0,
            months_behind_coalition=self._get_months_behind(nation_type),
            base_ai_research_talent=self._get_ai_research_talent(nation_type),
            threat_level=self._assign_threat_level(nation_type)
        )
    
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
        
        # Post-process: Update AI R&D multipliers based on talent and hardware
        self._update_rd_multipliers(all_actors)
        
        return all_actors
    
    def _update_rd_multipliers(self, actors: List[Actor]):
        """Update AI R&D multipliers based on talent and hardware levels"""
        for actor in actors:
            # Base multiplier from talent
            talent_multiplier = actor.base_ai_research_talent
            
            # Hardware boost (more compute = faster research)
            hardware_boost = 1.0 + np.log10(max(1, actor.compute_h100e / 1000)) * 0.1
            
            # Coalition penalty/boost
            coalition_multiplier = 1.2 if actor.is_coalition_member else 0.8
            
            actor.ai_rd_prog_multiplier = talent_multiplier * hardware_boost * coalition_multiplier
    
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
            'purchasing_power_billion_usd': actor.purchasing_power_billion_usd,
            'ai_rd_prog_multiplier': actor.ai_rd_prog_multiplier,
            'months_behind_coalition': actor.months_behind_coalition,
            'base_ai_research_talent': actor.base_ai_research_talent,
            'threat_level': actor.threat_level,
            'is_sanctioned': actor.is_sanctioned,
            'is_coalition_member': actor.is_coalition_member
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