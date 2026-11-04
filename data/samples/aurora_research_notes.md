---
project: Aurora
owner: Dr. Priya Raman
updated: 2026-03-10
---

# Project Aurora: Research Notes

Project Aurora is Helios Energy Lab's program to develop a low-cost sodium-ion cell for
stationary storage. The target is **80% capacity retention after 3000 cycles** at 25 °C.

## Cell Chemistry

The cathode is a Prussian white (Na₂Fe[Fe(CN)₆]) material; the anode is hard carbon derived
from coconut shell. Prussian white was chosen over layered oxides because it avoids nickel
and cobalt entirely.

- Cathode: Prussian white, 2.1 mAh/cm² loading
- Anode: hard carbon, N/P ratio 1.10
- Separator: ceramic-coated polyethylene, 16 µm

### Electrolyte Baselines

Three electrolyte formulations are under test:

1. **E-5**: 1.0 M NaPF₆ in EC:DEC (1:1), the original baseline
2. **E-5 + FEC**: E-5 with 2% fluoroethylene carbonate additive
3. **E-7**: 1.0 M NaPF₆ in EC:PC:DMC with 1% sodium difluoro(oxalato)borate

> FEC forms a more stable solid-electrolyte interphase on hard carbon, which is why it
> improves retention at room temperature.

## Results (as of 2026-03-10)

| Cell | Electrolyte | Temp (°C) | Cycles | Retention (%) |
|------|-------------|-----------|--------|---------------|
| AUR-C101 | E-5 + FEC | 25 | 1000 | 91.4 |
| AUR-C102 | E-5 | 25 | 1000 | 88.0 |
| AUR-C103 | E-5 | 45 | 412 | 72.5 |
| AUR-C104 | E-5 + FEC | 45 | 800 | 83.1 |

Cell AUR-C103 failed at cycle 412 because of gas evolution at 45 °C; the pouch swelled by
14%. Elevated-temperature stability of E-5 is therefore the main open risk.

## Test Configuration

```yaml
protocol: CC-CV
charge_rate: 0.5C
discharge_rate: 1C
voltage_window: [1.5, 3.9]
checkpoint_every: 100
```

![Capacity retention curves for AUR-C101 to AUR-C104](figures/retention_curves.png)
