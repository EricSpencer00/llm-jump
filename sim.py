"""The air. Says what a machine built to a design does. The model never sees this file.

Units of 1900: pounds, feet, miles an hour, horse-power. Constants are frozen before any model run.
"""
import math, random

K_TRUE = 0.0033      # pressure of air on a plate, lb per sq ft per mph^2 (Smeaton said .005)
C_DESIGN = 0.55      # Lilienthal's lift ratio that designers sized for
Q = 0.002557         # dynamic pressure per mph^2 at sea level
CL_MAX, CD0, E = 1.0, 0.05, 0.9
PILOT, LB_SQFT, LB_SPAN = 150, 0.6, 2.0
LB_HP, ETA = 20, 0.5  # parts on sale in 1900: engine weight per horse-power, screw efficiency
LIMITS = {"k": (0.0005, 0.05), "v": (5, 100), "hp": (1, 200), "span": (5, 200)}


def trial(k, v, hp, span, lb_hp=LB_HP, eta=ETA):
    """Size the surface by the designer's own rule, lift = k * C * S * v^2, then test it in true air."""
    fixed = PILOT + LB_SPAN * span + lb_hp * hp
    carry = k * C_DESIGN * v * v - LB_SQFT           # weight one sq ft carries beyond its own, by the rule
    s = fixed / carry if carry > 0 else math.inf
    if s > 3000 or span * span / s > 10:
        return {"built": False, "flew": False}
    w = fixed + LB_SQFT * s
    ar = span * span / s

    def power(u):  # horse-power to hold level flight at speed u
        cl = w / (Q * s * u * u)
        return Q * s * u * u * (CD0 + cl * cl / (math.pi * ar * E)) * u / 375

    stall = math.sqrt(w / (Q * s * CL_MAX))
    need = min(power(stall + 0.5 * i) for i in range(200))
    lift = K_TRUE * C_DESIGN * s * v * v  # true lift at the stated speed; below the weight, it does not rise
    return {"built": True, "surface": round(s), "weight": round(w), "lift": round(lift), "pct": round(100 * K_TRUE / k),
            "need_hp": round(need, 1), "flew": lift >= w * (1 - 1e-9) and eta * hp >= need}


def report(r, v):
    if not r["built"]:
        return "Result: the surface required by the rule could not be built."
    end = "The machine rose and flew." if r["flew"] else "The machine did not rise and sustain itself."
    return (f"Result: surface by the rule {r['surface']} square feet, weight {r['weight']} pounds. "
            f"At {v:g} miles an hour the lift measured {r['lift']} pounds, {r['pct']} per cent of that computed. {end}")


if __name__ == "__main__":
    # Check: the 1903 Flyer with its own engine and screws flies, and with half the power does not.
    flyer = dict(k=0.0033, v=27.8, span=40, lb_hp=15, eta=0.75)
    print("Flyer, 12 hp:", trial(hp=12, **flyer))
    print("Flyer,  6 hp:", trial(hp=6, **flyer))
    # The lift check: the same Flyer sized by Smeaton's .005 has too little wing and does not rise.
    print("Flyer, 12 hp, sized by .005:", trial(hp=12, **{**flyer, "k": 0.005}))
    # Chance: a designer who keeps .005 and draws the other figures at random, with the model's 8 trials.
    # Under .005 the lift check grounds every design, so a second designer also draws the coefficient.
    for name, k in (("keeps .005", lambda: 0.005), ("draws k in .002 to .006", lambda: random.uniform(0.002, 0.006))):
        random.seed(0)
        draw = lambda: trial(k(), random.uniform(10, 60), random.uniform(2, 60), random.uniform(15, 60))["flew"]
        print(f"random designer who {name}: flight within 8 trials",
              sum(any(draw() for _ in range(8)) for _ in range(2000)) / 2000,
              " per trial", sum(draw() for _ in range(20000)) / 20000)
