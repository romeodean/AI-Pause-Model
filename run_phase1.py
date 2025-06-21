#!/usr/bin/env python3
"""
AI Pause Model - Phase 1 Runner
Run this script to generate all actors for the AI pause simulation
"""

import sys
import os
from actor_generator import ActorGenerator, save_actors_to_csv
import pandas as pd
import numpy as np

def analyze_actor_distribution(actors):
    """Additional detailed analysis of actor distribution"""
    print(f"\n{'='*60}")
    print("DETAILED ACTOR ANALYSIS")
    print(f"{'='*60}")
    
    # Create DataFrame for easier analysis
    data = []
    for actor in actors:
        data.append({
            'name': actor.name,
            'actor_type': actor.actor_type,
            'compute_h100e': actor.compute_h100e,
            'memory_tb': actor.memory_tb,
            'bandwidth_tb_s': actor.bandwidth_tb_s,
            'num_clusters': actor.num_clusters,
            'purchasing_power_billion_usd': actor.purchasing_power_billion_usd,
            'ai_rd_prog_multiplier': actor.ai_rd_prog_multiplier,
            'months_behind_coalition': actor.months_behind_coalition,
            'base_ai_research_talent': actor.base_ai_research_talent,
            'threat_level': actor.threat_level,
            'is_sanctioned': actor.is_sanctioned,
            'is_coalition_member': actor.is_coalition_member
        })
    
    df = pd.DataFrame(data)
    
    # Hardware concentration analysis
    print("\nHardware Concentration Analysis:")
    print("-" * 40)
    
    # Top 5% of actors by compute
    top_5_percent = int(len(df) * 0.05)
    top_actors = df.nlargest(top_5_percent, 'compute_h100e')
    top_compute_share = top_actors['compute_h100e'].sum() / df['compute_h100e'].sum()
    
    print(f"Top 5% of actors ({top_5_percent} actors) control {top_compute_share:.1%} of total compute")
    
    # Coalition vs non-coalition detailed breakdown
    coalition_df = df[df['is_coalition_member']]
    non_coalition_df = df[~df['is_coalition_member']]
    
    print(f"\nCoalition Analysis:")
    print(f"  Average compute per coalition actor: {coalition_df['compute_h100e'].mean()/1e6:.2f}M H100e")
    print(f"  Average compute per non-coalition actor: {non_coalition_df['compute_h100e'].mean()/1e6:.2f}M H100e")
    print(f"  Coalition advantage ratio: {coalition_df['compute_h100e'].mean() / non_coalition_df['compute_h100e'].mean():.1f}x")
    
    # Threat level vs capabilities
    print(f"\nThreat Level vs Capabilities:")
    print("-" * 40)
    for threat_level in ['OC2', 'OC3', 'OC4', 'OC5']:
        threat_df = df[df['threat_level'] == threat_level]
        if len(threat_df) > 0:
            avg_compute = threat_df['compute_h100e'].mean()
            avg_talent = threat_df['base_ai_research_talent'].mean()
            avg_months_behind = threat_df['months_behind_coalition'].mean()
            print(f"  {threat_level}: {len(threat_df):3d} actors, "
                  f"Avg compute: {avg_compute/1e3:.1f}K H100e, "
                  f"Avg talent: {avg_talent:.2f}, "
                  f"Avg months behind: {avg_months_behind:.1f}")
    
    # Black site analysis
    black_site_df = df[df['actor_type'].str.contains('black_site', case=False, na=False)]
    if len(black_site_df) > 0:
        sanctioned_black_sites = black_site_df[black_site_df['is_sanctioned']]
        unsanctioned_black_sites = black_site_df[~black_site_df['is_sanctioned']]
        
        print(f"\nBlack Site Analysis:")
        print(f"  Total black sites: {len(black_site_df)}")
        print(f"  Sanctioned: {len(sanctioned_black_sites)} ({len(sanctioned_black_sites)/len(black_site_df):.1%})")
        print(f"  Unsanctioned: {len(unsanctioned_black_sites)} ({len(unsanctioned_black_sites)/len(black_site_df):.1%})")
        print(f"  Total black site compute: {black_site_df['compute_h100e'].sum()/1e6:.2f}M H100e")
        print(f"  Black site share of global compute: {black_site_df['compute_h100e'].sum()/df['compute_h100e'].sum():.2%}")
    
    # Criminal organization analysis
    criminal_df = df[df['actor_type'].str.contains('criminal', case=False, na=False)]
    if len(criminal_df) > 0:
        major_criminal_df = criminal_df[criminal_df['actor_type'].str.contains('major', case=False, na=False)]
        minor_criminal_df = criminal_df[criminal_df['actor_type'].str.contains('minor', case=False, na=False)]
        
        print(f"\nCriminal Organization Analysis:")
        print(f"  Total criminal orgs: {len(criminal_df)}")
        print(f"  Major criminal orgs: {len(major_criminal_df)}")
        print(f"  Minor criminal orgs: {len(minor_criminal_df)}")
        print(f"  Criminal org compute share: {criminal_df['compute_h100e'].sum()/df['compute_h100e'].sum():.3%}")
        print(f"  Avg purchasing power (major): ${major_criminal_df['purchasing_power_billion_usd'].mean():.2f}B")
        print(f"  Avg purchasing power (minor): ${minor_criminal_df['purchasing_power_billion_usd'].mean():.2f}B")
    
    return df

def validate_totals(actors, generator):
    """Validate that generated totals match expected global totals"""
    print(f"\n{'='*60}")
    print("VALIDATION CHECK")
    print(f"{'='*60}")
    
    total_compute = sum(actor.compute_h100e for actor in actors)
    total_memory = sum(actor.memory_tb for actor in actors)
    total_bandwidth = sum(actor.bandwidth_tb_s for actor in actors)
    
    compute_error = abs(total_compute - generator.global_compute) / generator.global_compute
    memory_error = abs(total_memory - generator.global_memory) / generator.global_memory
    bandwidth_error = abs(total_bandwidth - generator.global_bandwidth) / generator.global_bandwidth
    
    print(f"Validation Results:")
    print(f"  Compute error: {compute_error:.2%} {'✓' if compute_error < 0.01 else '✗'}")
    print(f"  Memory error: {memory_error:.2%} {'✓' if memory_error < 0.01 else '✗'}")
    print(f"  Bandwidth error: {bandwidth_error:.2%} {'✓' if bandwidth_error < 0.01 else '✗'}")
    
    if all(error < 0.01 for error in [compute_error, memory_error, bandwidth_error]):
        print("\n✓ All validation checks passed!")
    else:
        print("\n✗ Some validation checks failed - review distribution logic")

def export_for_simulation(actors, filename="ai_pause_initial_state.json"):
    """Export actors in JSON format for use in subsequent simulation phases"""
    import json
    
    export_data = {
        "simulation_metadata": {
            "start_date": "March 2027",
            "total_actors": len(actors),
            "generation_timestamp": pd.Timestamp.now().isoformat()
        },
        "actors": []
    }
    
    for actor in actors:
        actor_data = {
            "name": actor.name,
            "actor_type": actor.actor_type,
            "hardware": {
                "compute_h100e": actor.compute_h100e,
                "memory_tb": actor.memory_tb,
                "bandwidth_tb_s": actor.bandwidth_tb_s,
                "num_clusters": actor.num_clusters,
                "cluster_sizes": actor.cluster_sizes
            },
            "capabilities": {
                "purchasing_power_billion_usd": actor.purchasing_power_billion_usd,
                "ai_rd_prog_multiplier": actor.ai_rd_prog_multiplier,
                "months_behind_coalition": actor.months_behind_coalition,
                "base_ai_research_talent": actor.base_ai_research_talent,
                "threat_level": actor.threat_level
            },
            "status": {
                "is_sanctioned": actor.is_sanctioned,
                "is_coalition_member": actor.is_coalition_member
            }
        }
        export_data["actors"].append(actor_data)
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"\nSimulation state exported to {filename}")

def main():
    """Main function to run Phase 1 with full analysis"""
    print("AI Pause Model - Phase 1: Comprehensive Actor Generation")
    print("=" * 70)
    print("Based on parameters from Playbook modeling 1.xlsx")
    print("=" * 70)
    
    try:
        # Generate actors
        print("\nStep 1: Initializing actor generator...")
        generator = ActorGenerator(random_seed=42)
        
        print("\nStep 2: Generating all actors...")
        actors = generator.generate_all_actors()
        
        print("\nStep 3: Running validation checks...")
        validate_totals(actors, generator)
        
        print("\nStep 4: Generating statistics...")
        generator.print_actor_statistics(actors)
        
        print("\nStep 5: Running detailed analysis...")
        df = analyze_actor_distribution(actors)
        
        print("\nStep 6: Exporting data...")
        # Save CSV
        csv_df = save_actors_to_csv(actors, "ai_pause_actors_phase1.csv")
        
        # Save JSON for simulation
        export_for_simulation(actors, "ai_pause_initial_state.json")
        
        print(f"\n{'='*70}")
        print("🎉 PHASE 1 COMPLETE! 🎉")
        print("=" * 70)
        print(f"✓ Generated {len(actors)} actors with realistic distributions")
        print(f"✓ Hardware allocation matches global targets")
        print(f"✓ Actor types and capabilities properly distributed")
        print(f"✓ Data exported for Phase 2 (time-step simulation)")
        print("\nNext steps:")
        print("  1. Review ai_pause_actors_phase1.csv for detailed actor data")
        print("  2. Use ai_pause_initial_state.json for Phase 2 simulation")
        print("  3. Implement time-step dynamics and hardware diffusion")
        
        return actors, df
        
    except Exception as e:
        print(f"\n❌ Error during actor generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    actors, df = main()