def comfort_risk(ambient_c: float, cabin_c: float, setpoint_delta: int):
    """
    Returns (label, color) where color is a simple tag for Streamlit.
    Logic: base on ΔT = cabin - ambient, and whether user is moving the setpoint
    further from a comfortable band (22–24°C).
    """
    dT = cabin_c - ambient_c
    moving_cooler = setpoint_delta < 0
    moving_warmer = setpoint_delta > 0

    # Ambient bands
    if ambient_c >= 28:
        # Hot outside: going warmer saves energy with low comfort risk; going cooler increases risk
        if moving_cooler and abs(setpoint_delta) >= 3:
            return ("High risk: already hot, cooler setpoint may be uncomfortable", "red")
        if moving_warmer and setpoint_delta >= 2:
            return ("Low risk: warmer setpoint likely fine in heat", "green")
        return ("Med risk: balance comfort vs savings", "orange")
    elif ambient_c <= 12:
        # Cold outside: warmer setpoint OK; cooler increases discomfort
        if moving_cooler and abs(setpoint_delta) >= 2:
            return ("Med–High risk: cooler setpoint in cold may feel harsh", "red")
        if moving_warmer and setpoint_delta >= 2:
            return ("Low risk: warmer setpoint improves comfort", "green")
        return ("Low–Med risk", "orange")
    else:
        # Mild
        if abs(setpoint_delta) <= 2:
            return ("Low risk: mild weather", "green")
        return ("Med risk", "orange")

def effect_hint(ambient_c: float, recirc_on: bool):
    if ambient_c >= 28:
        return "Hot outside: Recirc ON saves more; consider +1–2°C setpoint."
    if ambient_c <= 12:
        return "Cold outside: Warmer setpoint (+1–2°C) reduces heater load."
    return "Mild weather: small changes have small effects."
