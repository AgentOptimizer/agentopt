"""Reusable tools for MathQA benchmark agents."""

from __future__ import annotations

import math
import re

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Supports standard arithmetic (+, -, *, /, //, %, **), math functions
    (sqrt, sin, cos, tan, log, log10, ceil, floor, abs, round, factorial,
    pi, e), and parentheses. Examples: "2 + 3 * 4", "sqrt(144)", "15 % 4".
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "ceil": math.ceil,
        "floor": math.floor,
        "abs": abs,
        "round": round,
        "pow": pow,
        "factorial": math.factorial,
        "pi": math.pi,
        "e": math.e,
        "min": min,
        "max": max,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def solve_equation(equation: str) -> str:
    """Solve a simple linear equation for x and return the result.

    The equation must contain exactly one unknown 'x'.
    Examples: "2*x + 3 = 11", "x/4 = 5", "100 - x = 37".
    """
    equation = equation.strip()
    if "=" not in equation:
        return "Error: equation must contain '='"
    if "x" not in equation.lower():
        return "Error: equation must contain variable 'x'"

    lhs, rhs = equation.split("=", 1)

    allowed = {
        "sqrt": math.sqrt,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
        "__builtins__": {},
    }

    def f(x_val: float) -> float:
        env = {**allowed, "x": x_val, "X": x_val}
        return float(eval(lhs, env)) - float(eval(rhs, env))

    try:
        f0 = f(0.0)
        f1 = f(1.0)
        a = f1 - f0
        if abs(a) > 1e-12:
            x_solution = -f0 / a
            if abs(f(x_solution)) < 1e-6:
                if abs(x_solution - round(x_solution)) < 1e-9:
                    return str(int(round(x_solution)))
                return str(x_solution)

        lo, hi = -1e6, 1e6
        if f(lo) * f(hi) > 0:
            return f"Error: could not bracket a solution in [{lo}, {hi}]"
        for _ in range(100):
            mid = (lo + hi) / 2
            if abs(f(mid)) < 1e-9:
                break
            if f(lo) * f(mid) < 0:
                hi = mid
            else:
                lo = mid
        result = (lo + hi) / 2
        if abs(result - round(result)) < 1e-6:
            return str(int(round(result)))
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between common units.

    Supported categories:
    - Length: km, m, cm, mm, mi, yd, ft, in
    - Weight: kg, g, mg, lb, oz
    - Time: hr, min, sec
    - Speed: km/h, m/s, mph
    - Temperature: C, F, K
    - Volume: L, mL, gal

    Examples: value=5, from_unit="km", to_unit="mi"
    """
    length_to_m = {
        "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001,
        "mi": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254,
    }
    weight_to_kg = {
        "kg": 1, "g": 0.001, "mg": 1e-6, "lb": 0.453592, "oz": 0.0283495,
    }
    time_to_sec = {"hr": 3600, "min": 60, "sec": 1}
    speed_to_ms = {"km/h": 1 / 3.6, "m/s": 1, "mph": 0.44704}
    volume_to_l = {"L": 1, "mL": 0.001, "gal": 3.78541}

    from_unit = from_unit.strip()
    to_unit = to_unit.strip()

    temp_units = {"C", "F", "K"}
    if from_unit in temp_units and to_unit in temp_units:
        if from_unit == "F":
            c = (value - 32) * 5 / 9
        elif from_unit == "K":
            c = value - 273.15
        else:
            c = value
        if to_unit == "F":
            result = c * 9 / 5 + 32
        elif to_unit == "K":
            result = c + 273.15
        else:
            result = c
        return str(round(result, 6))

    for table in [length_to_m, weight_to_kg, time_to_sec, speed_to_ms, volume_to_l]:
        if from_unit in table and to_unit in table:
            base_value = value * table[from_unit]
            result = base_value / table[to_unit]
            if abs(result - round(result)) < 1e-9:
                return str(int(round(result)))
            return str(round(result, 6))

    return f"Error: cannot convert from '{from_unit}' to '{to_unit}'"


TOOLS = [calculator, solve_equation, unit_converter]
