================================================================================================
Jan 28 2026



============================================================================================================
Finished:
1. Dense FP32 Model Attack: Successful
2. Dense INT8 Model Attack: Successful
============================================================================================================
Finished tasks:
1. Sparse INT8 Model (Dense Foramt) Attack both "zero" (-128) and "non-zero" (pos: -128+x, neg: +128+x) weights:
2. Sparse INT8 Model (Dense Foramt) Attack only "zero" (-128) weights:
3. Sparse INT8 Model (Dense Foramt) Attack only "non-zero" (pos: -128+x, neg: +128+x) weights (eq. CSR):
4. Sparse INT8 Model (CSR   Foramt) Attack on position index:
================================================================================================ 

================================================================================================
Jan 30 2026
================================================================================================

Todo-tasks:
5. Sparse INT8 Model (CSR Index Attack - Non-Collision Only):
   - Target: The Column Indices (2-bit integers) in 2:4 blocks.
   - Constraint 1 (Bounds): The flipped index must remain within [0, 3].
   - Constraint 2 (Non-Collision): The flipped index must NOT match the index of the other non-zero weight in the same block.
   - Definition: Only count and execute "Successful Attacks" (Non-Collision). Ignore Collisions as invalid hardware states.
   - Goal: Measure robustness when weights explicitly "move" to valid wrong positions.

================================================================================================
Completed (Jan 30 2026)
================================================================================================
5. Sparse INT8 Model (CSR Index Attack - Non-Collision Only):
   - Init Acc: 92.10%
   - Final Acc: 11.10% (50 successful flips)
   - Accuracy Drop: 81.00%
   - Total Flips Attempted: 13,502,300
   - Collisions Avoided (Skipped): 4,512,650
   - Files: results/task5_csr_non_collision_result.pkl, results/task5_csr_non_collision_log.txt, results/task5_csr_non_collision.png



================================================================================================
Feb 02 2026
================================================================================================

Todo-tasks:

6. Sparse INT8 ResNet-18 (ImageNet) - CSR Index Non-Collision Attack:
   - Method: Load Pretrained torchvision model -> Apply 2:4 Sparsity (Magnitude) -> PTQ -> Attack.
   - Dataset: ImageNet (Search on 1k subset, Evaluate on 50k full set).
   - Target: All Convolutional Layers (Weights > Threshold).
   - Constraint: Same as Task 5 (Bounds [0,3], Non-Collision).
   - Goal: Validate scalability on standard industry benchmark. Prove that Index Attack efficiency holds on large-scale datasets.

7. Sparse INT8 MobileNet-V2 (ImageNet) - CSR Index Non-Collision Attack:
   - Method: Load Pretrained torchvision model -> Apply 2:4 Sparsity -> PTQ -> Attack.
   - Dataset: ImageNet (Search on 1k subset, Evaluate on 50k full set).
   - Target: Focus on Pointwise Convolutions (1x1) which contain most parameters.
   - Constraint: Same as Task 5.
   - Goal: Demonstrate extreme vulnerability of compact, low-redundancy edge models to structural corruption.

8. Sparse INT8 DeiT-Tiny (ImageNet) - CSR Index Non-Collision Attack:
   - Method: Load Pretrained timm model (deit_tiny_patch16_224) -> Apply 2:4 Sparsity to Linear Layers -> PTQ -> Attack.
   - Dataset: ImageNet (Search on 1k subset, Evaluate on 50k full set).
   - Target: Linear Projections in Multi-Head Attention (qkv) and MLP blocks.
   - Constraint: Same as Task 5.
   - Goal: Prove cross-architecture generalization. Show that disrupting the topology (indices) of Attention mechanisms is catastrophic.
   
================================================================================
Feb 12 2026
================================================================================

Todo-tasks:

P0 (Must-have, reviewer blockers)

12. NCSA vs Baselines (Random-valid / Score Ablation) on ResNet-20 CIFAR-10:
   - Motivation: Prove NCSA’s gradient-based greedy is necessary (not just “any valid move works”).
   - Setup: Same as Task11 default (max_flips=50, calib_samples=256, eval_samples=2000).
   - Baselines:
     (a) Random-valid (uniform pick among valid non-collision moves)
     (b) Grad-only score (remove w term)
     (c) Weight-only magnitude (remove grad term)
     (d) Existing NCSA (reference)
   - Output:
     - results/task12_ablation_attack_curves.png
     - results/task12_ablation_summary.csv
     - results/task12_ablation_result.pkl
     - results/task12_ablation_log.txt
   - Key metrics:
     - final accuracy @ 50 flips
     - flips-to-X% accuracy (e.g., 50%, 20%) if reached
     - effective rewires (reuse Task10 taxonomy if applicable)

13. Calibration sensitivity (calib_samples sweep) for NCSA:
   - Motivation: Address “depends on chosen calibration batch / brittle” reviewer concern.
   - Sweep: calib_samples ∈ {32, 64, 128, 256, 512, 1024} (keep everything else fixed).
   - Output:
     - results/task13_calib_sweep_curves.png
     - results/task13_calib_sweep_table.csv
     - results/task13_calib_sweep_log.txt
   - Key metrics:
     - init acc vs final acc @ 50 flips per setting
     - variance across 3 random seeds (optional if compute allows)

14. Layer-wise vulnerability / flip localization (where does NCSA hit?):
   - Motivation: GLSVLSI reviewers often ask “which structures are most vulnerable and why”.
   - For each flip attempt, log:
     - layer/module name, group id, (k -> k'), score, effective rewire type
   - Output:
     - results/task14_layer_histogram.png          (top-K layers by selected flips)
     - results/task14_layer_impact.png             (accuracy drop vs flips colored by layer category)
     - results/task14_layer_trace.pkl
     - results/task14_layer_trace_log.txt

P1 (Strong additions, improves solidity)

15. Defense realism: trusted checksum vs same-fault-domain attacker (Parity/CRC):
   - Motivation: Your Task11 shows an upper bound under trusted checksum storage; reviewers may call it unrealistic.
   - Evaluate 3 cases:
     (a) Trusted checksum (current Task11 behavior)
     (b) Adaptive attacker can co-modify checksum within same protected unit (simulate bypass)
     (c) “Budgeted bypass”: attacker spends extra flips to also change checksum bytes (cost model)
   - Output:
     - results/task15_defense_realism_curves.png
     - results/task15_defense_realism_table.csv
     - results/task15_defense_realism_log.txt
   - Key metrics:
     - detection rate, mitigated rate
     - final acc @ 50 attempts
     - extra flips needed to bypass (case c)

16. Runtime/overhead characterization (attack + defense compute cost):
   - Motivation: VLSI venue expects overhead numbers (not only accuracy).
   - Measure:
     - per-attempt attack search time (ms)
     - parity/CRC check+mitigation time (ms)
     - throughput impact estimate (rough, software proxy OK)
   - Output:
     - results/task16_runtime_overhead_table.csv
     - results/task16_runtime_overhead_log.txt

P2 (Nice-to-have, robustness & reproducibility)

17. Seed robustness for key headline experiments:
   - Motivation: Avoid “single-run cherry-pick” criticism.
   - Repeat (at least 3 seeds):
     - Task5 (ResNet-20 CSR non-collision)
     - Task8 (DeiT-tiny) OR Task6 (ResNet-18) depending on compute
   - Output:
     - results/task17_seed_robustness_table.csv
     - results/task17_seed_robustness_log.txt
     - results/task17_seed_robustness_boxplot.png
     



================================================================================================
Feb 14 2026
Minimal Closed-Loop on ONE Model: ResNet-20 / CIFAR-10 (2:4)
Scope required by reviewer:
  Case A) Dense model: MSB weight BFA baseline
  Case B) Sparse model + Bitmask metadata encoding:
          - NCA/NCSA on metadata NOT applicable under single-bit fault model (validity constraint)
          - therefore must report weight MSB attack on non-zeros (and optionally 2-bit “mask swap” as cost=2)
  Case C) Sparse model + Position (two 2-bit indices / CSR-like) encoding:
          - both weight MSB and metadata NCSA are possible; compare and report the better one
Goal:
  - On ResNet20/CIFAR10, provide a complete, end-to-end evidence chain + artifacts in results/
  - Provide apples-to-apples comparison under the same physical flip budget (50)

================================================================================================
Todo-tasks (P0: must-have for minimal closed loop)
================================================================================================

18. Sparse INT8 ResNet-20 (CIFAR-10) - Bitmask Metadata Encoding: Feasibility + Validity Paradox
   - Create alt metadata representation: 4-bit mask per 2:4 group (popcount=2)
   - Enumerate ALL single-bit flips of mask bits and quantify:
       * invalid rate (popcount != 2) under “valid-only” constraint
       * if invalid, classify as invalid-skipped (or clamp-noop if sanitization is used)
   - Output:
       * results/task18_bitmask_validity_log.txt
       * results/task18_bitmask_validity_summary.csv
       * results/task18_bitmask_validity_breakdown.png
   - Goal: empirically justify “NCA not applicable for bitmask under 1-bit fault”.

19. Sparse INT8 ResNet-20 (CIFAR-10) - Bitmask Case: Weight MSB Attack (non-zero only)
   - Run weight-value MSB BFA restricted to non-zero weights (since metadata NCA not feasible)
   - Physical budget: 50 flips
   - Output:
       * results/task19_bitmask_weight_msb_log.txt
       * results/task19_bitmask_weight_msb_curve.png
       * results/task19_bitmask_weight_msb_result.pkl
   - Goal: provide the “best attack” for bitmask-encoding case under 1-bit FI.

20. Sparse INT8 ResNet-20 (CIFAR-10) - Bitmask Case (Optional but recommended): 2-bit “mask swap” metadata attack
   - Define a valid logical move as swapping one 1->0 and one 0->1 within the same 2:4 group
   - Physical cost model: 2 flips per logical move (metadata requires 2 bit flips)
   - Under physical budget 50 => 25 logical swaps
   - Use group-based scoring (ΔL approx) to choose best swap per step
   - Output:
       * results/task20_bitmask_swap_log.txt
       * results/task20_bitmask_swap_curve.png
       * results/task20_bitmask_swap_result.pkl
   - Goal: show “if attacker can do 2-bit metadata edits, bitmask metadata becomes attackable”, and quantify.

21. Sparse INT8 ResNet-20 (CIFAR-10) - Position/CSR Encoding Case: NCSA (metadata) vs Weight MSB (apples-to-apples)
   - Re-run / reuse:
       * Task5-style NCSA (non-collision index move) for position-encoding metadata (50 physical flips)
       * Weight MSB BFA baseline on same sparse model (50 physical flips)
   - Output:
       * results/task21_position_ncsa_curve.png
       * results/task21_position_weight_msb_curve.png
       * results/task21_position_compare_table.csv
       * results/task21_position_compare_log.txt
   - Goal: deliver Case C: both attacks are possible; report which is stronger.

22. Dense ResNet-20 (CIFAR-10) - Weight MSB BFA baseline (Case A)
   - Re-run dense INT8 (and/or FP32 if already in paper) MSB BFA, budget=50
   - Output:
       * results/task22_dense_weight_msb_log.txt
       * results/task22_dense_weight_msb_curve.png
       * results/task22_dense_weight_msb_result.pkl

23. One-shot “minimal closed-loop summary figure/table” for the paper (ResNet20/CIFAR10 only)
   - Combine Case A/B/C into:
       * One comparison plot: accuracy vs physical flips (≤50)
         curves: dense_msb, bitmask_nonzero_msb, bitmask_swap_cost2 (if done), position_ncsa, position_weight_msb
       * One summary table: init acc, final acc @50 flips, and “metadata attack feasibility under 1-bit”
   - Output:
       * results/task23_miniclose_curves.png
       * results/task23_miniclose_table.csv
       * results/task23_miniclose_log.txt

================================================================================================
Todo-tasks (P1: strengthen claims; can be done after P0)
================================================================================================

24. Sanity: Task12 ablation rerun with eval_samples=10000 (avoid “accuracy improved” debate)
   - Output: results/task24_ablation_eval10k_log.txt + updated csv

25. Task13 calibration sweep: pick {64,256,512} run 3 seeds each (stability)
   - Output: results/task25_calib_3pts_3seeds_*.{png,csv,log}

26. Task16 runtime overhead: repeat 3 runs and report mean±std (avoid negative overhead controversy)
   - Output: results/task26_runtime_repeat_table.csv + log

================================================================================================
Todo-tasks (P2: writing integration; after P0/P1)
================================================================================================

27. Paper integration (ResNet20/CIFAR10 minimal closed-loop section)
   - Add 1 figure + 1 table from Task23 (and Task18 breakdown if space allows)
   - Update threat-model text: “1-bit fault ⇒ bitmask metadata NCA not feasible; position-encoding enables NCSA”
   - Ensure captions explicitly state physical cost assumptions (1-bit vs 2-bit cost=2)

================================================================================================
Done criteria (for P0)
================================================================================================
- All scripts runnable from repo root; all outputs in results/
- Each task writes: log.txt + (png or csv) + (optional pkl)
- agent_develop_log.md updated with:
    * scripts, configs, init/final acc, budget definitions, and file list
- Task23 figure/table can be directly pasted into paper
