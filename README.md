# Multi-Armed Bandits for Adaptive Bit Rates

1. Install Dependencies First:
   
    ``` bash
    pip install -r requirements.txt
    ```

2. Run Options:

    Run a single algorithm:

    ``` bash

    python mab_tester.py <algorithm_name>
    ```

    Run ALL algorithms and generate comparison plots:

    ``` bash
    python mab_tester.py ALL

    This will:

    Run all algorithms found in the student/ folder
    Execute each algorithm 10 times per test scenario (N_TRIALS = 10)
    Generate heatmaps, CDFs, boxplots, and learning curves comparing all algorithms