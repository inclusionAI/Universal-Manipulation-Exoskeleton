import numpy as np

def tau_friction_compensation(qvel, coeff, max_comp):
    tau_friction = np.clip(coeff * qvel, -max_comp, max_comp)
    return tau_friction

def tau_stiction_compensation(qvel, sti_threshold_min, sti_threshold_max, sti_comp):
    tau_stiction = np.where((sti_threshold_min < np.abs(qvel)) & (np.abs(qvel) < sti_threshold_max), np.sign(qvel) * sti_comp, 0)
    return tau_stiction

