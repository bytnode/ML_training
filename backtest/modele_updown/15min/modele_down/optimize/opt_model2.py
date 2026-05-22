"""Combo regularization sweep -- the HP sweep showed the ETH model is
under-regularized (mcw 20->50 lifts honest older WR +0.60pt). Test mcw x
reg_lambda x reg_alpha combos on walk-forward, fixed calibration.
"""
import lib
import opt_model as om


def main():
    df = lib.load_master()
    fc = lib.feature_cols(df)
    print(f"features={len(fc)}\n")

    print("=== baseline (mcw=20, no reg) ===")
    om.score(df, fc, None, 4000, "baseline")

    print("\n=== regularization combos ===")
    for hp, tag in [
        ({"min_child_weight": 50}, "mcw50"),
        ({"min_child_weight": 65}, "mcw65"),
        ({"min_child_weight": 50, "reg_lambda": 3.0}, "mcw50 rl3"),
        ({"min_child_weight": 50, "reg_lambda": 5.0}, "mcw50 rl5"),
        ({"min_child_weight": 50, "reg_lambda": 8.0}, "mcw50 rl8"),
        ({"min_child_weight": 50, "reg_alpha": 1.0}, "mcw50 ra1"),
        ({"min_child_weight": 50, "reg_lambda": 5.0, "reg_alpha": 1.0}, "mcw50 rl5 ra1"),
        ({"min_child_weight": 65, "reg_lambda": 5.0}, "mcw65 rl5"),
        ({"min_child_weight": 80, "reg_lambda": 5.0}, "mcw80 rl5"),
        ({"min_child_weight": 50, "reg_lambda": 3.0, "gamma": 3.0}, "mcw50 rl3 g3"),
    ]:
        om.score(df, fc, hp, 4000, tag)


if __name__ == "__main__":
    main()
