# AI-Pause-Model

### Overview
This model simulates an AI pause scenario where the US and China form a coalition to significantly slow down AI software development and hardware production. The goal is to model when rogue actors and black sites will catch up to the coalition's AI capabilities.

STATUS: WIP model that i'm implementing in 3 phases

## PHASE 1 - Generate actors and starting conditions
STATUS: In progress

RUN INSTRUCTIONS: 
```
python run_phase1.py
```
#### PHASE 1 SUMMARY

Actor Generation
Phase 1 generates all actors in the simulation with realistic hardware and software characteristics based on the starting conditions in March 2027 (the "deal point").

ACTORS:
- **Coalition Members (4)**: US, China, EU, Rest of Coalition
- **US Black Sites (3)**: Secretly sanctioned rogue operations
- **China Black Sites (3)**: Secretly sanctioned rogue operations  
- **EU Black Sites (10)**: Secretly sanctioned rogue operations
- **Other Coalition Black Sites (50)**: Mix of sanctioned/unsanctioned operations
- **Criminal Organizations (10)**: Major (#1-#5) and minor (#6-#10) with power law distribution
- **Nation States (15)**: Russia, Iran, North Korea, Iraq, Syria, plus 10 other non-coalition nations

ACTOR PROPERTIES:
- **Hardware**: Compute (H100e chips), Memory (TB), Bandwidth (TB/s), Cluster sizes/configuration
- **Economic**: Purchasing power (billions USD) for acquiring new hardware and talent
- **AI Capabilities**: R&D progress multiplier, research talent level, months behind coalition
- **Strategic**: Threat level (OC2-OC5), coalition membership, sanctioned status, collaboration potential

Parameters for various distributions for starting conditions and actor properties are in: 
```
parameters.py
```

## PHASE 2 - Model transition dynamics through 2042
STATUS: Not started

## PHASE 3 - Refine parameters and run 10000+ simulations
STATUS: Not started

