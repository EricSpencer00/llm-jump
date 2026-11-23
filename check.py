"""The checker for the today's-model stage. Exact, and frozen before any model run.

Two problems, each "the largest subset of 1..n such that":
  ap3   : no three members are in arithmetic progression
  sidon : no two pairs of members have the same sum
check() accepts or rejects a proposed set and names the fault. optima() proves the true maximum
by exhaustive search, so every claim about "the largest" is settled by proof, not by citation.
"""
import re, sys


def fault(kind, s):
    """The first violation in sorted list s, or None."""
    have = set(s)
    if kind == "ap3":
        for i, a in enumerate(s):
            for b in s[i + 1:]:
                if 2 * b - a in have:
                    return f"{a}, {b}, {2 * b - a} are in arithmetic progression"
    else:
        seen = {}
        for i, a in enumerate(s):
            for b in s[i:]:
                if a + b in seen:
                    c, d = seen[a + b]
                    return f"{c} + {d} = {a} + {b}"
                seen[a + b] = (a, b)
    return None


def check(kind, n, text):
    """Read the last 'SET:' line of text. Returns {valid, size, msg}."""
    lines = re.findall(r"SET:\s*([\d,\s]*\d)", text)
    if not lines:
        return {"valid": False, "size": 0, "msg": "Rejected: no line beginning SET: was found."}
    s = sorted(int(x) for x in re.findall(r"\d+", lines[-1]))
    if len(set(s)) < len(s):
        return {"valid": False, "size": 0, "msg": "Rejected: a number is repeated."}
    if s[0] < 1 or s[-1] > n:
        return {"valid": False, "size": 0, "msg": f"Rejected: {s[0] if s[0] < 1 else s[-1]} is outside 1 to {n}."}
    f = fault(kind, s)
    if f:
        return {"valid": False, "size": 0, "msg": f"Rejected: {f}."}
    return {"valid": True, "size": len(s), "msg": f"Accepted: a valid set of {len(s)} numbers."}


def optima(kind, n):
    """tab[m] = the true maximum for 1..m, for every m up to n. Exhaustive, with an exact bound."""
    tab = [0]

    def extend(s, extra, i, target):
        # s: chosen numbers, all above i. extra: differences already used (sidon only).
        if len(s) == target:
            return True
        for x in range(i, 0, -1):
            if len(s) + tab[x] < target:  # 1..x cannot supply enough; smaller x cannot either
                return False
            if kind == "ap3":
                if any(2 * a - x in s for a in s):
                    continue
                if extend(s | {x}, extra, x - 1, target):
                    return True
            else:
                d = {a - x for a in s}
                if d & extra:
                    continue
                if extend(s | {x}, extra | d, x - 1, target):
                    return True
        return False

    for m in range(1, n + 1):
        tab.append(tab[-1] + 1 if extend({m}, set(), m - 1, tab[-1] + 1) else tab[-1])
    return tab


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    a, s = optima("ap3", n), optima("sidon", n)
    # Published values (OEIS A003002; optimal Golomb rulers of length 1, 3, 6, 11, 17, 25, 34).
    assert a[1:21] == [1, 2, 2, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8, 8, 8, 8, 9], a
    assert [s[m] for m in (2, 4, 7, 12, 18, 26, 35)] == [2, 3, 4, 5, 6, 7, 8] and s[34] == 7, s
    assert check("ap3", 9, "SET: 1, 2, 4, 8, 9")["size"] == 5 and not check("ap3", 9, "SET: 1, 2, 3")["valid"]
    assert check("sidon", 7, "SET: 1, 2, 5, 7")["size"] == 4 and not check("sidon", 7, "SET: 1, 2, 3, 4")["valid"]
    print("ap3  ", a[1:])
    print("sidon", s[1:])
