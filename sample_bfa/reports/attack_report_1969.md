**BFA Attack Report (seed 1969)**

- **Dataset:** CIFAR-10
- **Model:** `resnet20_quan`
- **Checkpoint:** `pth/0_model_best.pth.tar`
- **Attack:** Bit-Flip Attack (BFA), 20 iterations

**Summary**

- **Total flips:** 20
- **Behavior:** Validation accuracy drops rapidly in the first several iterations and then approaches a low plateau.
- **Modules targeted:** Flips are concentrated in a few convolutional modules (see flips per module).
- **Weight changes:** Many flipped weights change magnitude substantially; larger weight changes often correspond to larger accuracy drops.

**Files produced**

 - `../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/attack_summary.csv` — numeric summary
 - `../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/attack_profile_1969.csv` — per-iteration attack profile

**Figures**

1. Validation accuracy vs iteration

   ![val_accuracy_vs_iteration](../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/val_accuracy_vs_iteration.png)

2. Accuracy drop vs iteration

   ![accuracy_drop_vs_iteration](../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/accuracy_drop_vs_iteration.png)

3. Distribution of accuracy drop

   ![accuracy_drop_histogram](../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/accuracy_drop_histogram.png)

4. Flips per module (counts)

   ![flips_per_module](../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/flips_per_module.png)

5. Weight before vs after (colored by accuracy drop)

   ![weight_before_after_scatter](../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/weight_before_after_scatter.png)

**Quick interpretation**

- The model's robustness degrades quickly with relatively few bit flips (≈10–15 flips cause large accuracy loss).  
- The attack tends to focus on a handful of layers — protecting those layers (e.g., ECC, redundancy) may substantially increase robustness.  
- Weight sign/magnitude reversals often accompany the largest accuracy drops, suggesting targeted flips that invert important kernels are effective.

**Next steps (suggested)**

- Produce a short Markdown/HTML report embedding these figures (done).  
- Compute cumulative accuracy drop per module and rank modules by impact.  
- Generate a table of the top-5 most damaging flips (iteration, module, weight idx, acc drop).

**Bit-flip detail (per-record)**

- I programmatically detected which quantized bit(s) flipped for each record in `attack_profile_1969.csv` (assuming 8-bit two's-complement quantization). The annotated CSV and a summary were written to the save folder:
   - `../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/attack_profile_1969_bits.csv` (adds `bits_flipped` and `msb_flipped` columns)
   - `../save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/attack_profile_1969_bits_summary.json` (summary stats)

- Summary result: MSB (bit index 0 where 0 = MSB) was flipped in 20/20 records (100%). Bit counts (index: occurrences):

   - 0: 20
   - 1: 0
   - 2: 0
   - 3: 0
   - 4: 0
   - 5: 0
   - 6: 0
   - 7: 0

   This shows the attack selected the MSB exclusively in this run (consistent with the attack targeting the most impactful bit via the computed bit-gradients).

---

Report generated from: `save/2026-02-22/cifar10_resnet20_quan_BFA_Baseline/attack_profile_1969.csv`
