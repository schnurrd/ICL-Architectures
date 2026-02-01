# Attention Masking Modes

Controls visibility between **Train** (Context) and **Test** (Target) samples.
**Standard Setup:** `single_eval_pos` restricts keys to **Train Only**. Test samples *never* see other Test samples.

**Legend:**
*   `.` : Visible
*   `x` : Masked (or keys don't exist)
*   `Quadrants`: Top-Left (Train$\to), Bottom-Left (Test$\to). Right side is effectively void.

## 1. `None` 
**Train:** Bidirectional (sees all Train). **Test:** Sees all Train.
```text
       Keys (Train)  (Test)
       0 1 2   |     3 4
     +---------+-----------
  0  | . . .   |     x x
  1  | . . .   |     x x  (Train)
  2  | . . .   |     x x
     +---------+-----------
  3  | . . .   |     x x
  4  | . . .   |     x x  (Test)
```

## 2. `causal_train_only`
**Train:** Autoregressive. **Test:** Sees all Train.
```text
       Keys (Train)  (Test)
       0 1 2   |     3 4
     +---------+-----------
  0  | . x x   |     x x
  1  | . . x   |     x x
  2  | . . .   |     x x
     +---------+-----------
  3  | . . .   |     x x
  4  | . . .   |     x x
```

## 3. `test_to_train_only`
**Train:** Diagonal (Self-only). **Test:** Sees all Train.
```text
       Keys (Train)  (Test)
       0 1 2   |     3 4
     +---------+-----------
  0  | . x x   |     x x
  1  | x . x   |     x x
  2  | x x .   |     x x
     +---------+-----------
  3  | . . .   |     x x
  4  | . . .   |     x x
```

> **Feature Attention:** Orthogonal to Item Masking. Features attend fully to each other *within* the visible items defined above.
