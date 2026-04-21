# Meta-ABR Simulation Framework

CNT6885: Distributed Multimedia Systems — Spring 2026  
University of Florida, CISE Department  

## Overview

This repository contains a full simulation framework for evaluating adaptive bitrate (ABR) streaming algorithms under realistic network conditions.

It accompanies the term paper:

**"Adaptive Bitrate Streaming with Meta-Reinforcement Learning (Meta-ABR)"**

The simulator implements classical ABR baselines and a proposed Meta-ABR policy, and reproduces all experimental figures used in the paper.

---

## Implemented Algorithms

The following ABR algorithms are included:

1. Buffer-Based Adaptation (BBA) — Huang et al., SIGCOMM 2014  
2. RobustMPC — Yin et al., SIGCOMM 2015  
3. BOLA — Spiteri et al., INFOCOM 2016  
4. Pensieve (simplified RL approximation) — Mao et al., SIGCOMM 2017  
5. Meta-ABR (proposed meta-RL inspired policy)

---

## Features

- Synthetic but realistic network trace generator (4G / 5G / WiFi scenarios)
- QoE-driven evaluation framework
- Rebuffering and smoothness modeling
- Multi-scenario benchmarking
- Automatic figure generation for paper results

---

## Output Figures

Running the simulation generates:

- Fig. 1: Mean QoE across algorithms and scenarios  
- Fig. 2: Rebuffering ratio comparison  
- Fig. 3: Bitrate and buffer dynamics (Congested 4G)  
- Fig. 4: Adaptation behavior under regime shift  
- Fig. 5: QoE CDF across sessions  

---

## Requirements

```bash
pip install numpy matplotlib
