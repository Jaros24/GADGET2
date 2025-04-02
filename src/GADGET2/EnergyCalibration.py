"""
This module provides functions to convert ADC counts to MeV and vice versa
"""
# TODO: store calibration points alongside the run files. See github issue #1

def to_MeV(counts:float, calib_point_1:tuple[float, float]=(0.806, 156745), calib_point_2:tuple[float,float]=(1.679, 320842))->float:
    """
    Converts ADC counts to MeV using the energy calibration points.
    
    The calibration points are defined as tuples of (energy, channel).
    
    Returns the energy in MeV.
    """
    
    energy_1, channel_1 = calib_point_1
    energy_2, channel_2 = calib_point_2
    
    energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
    energy_offset = energy_1 - energy_scale_factor * channel_1
    
    return counts*energy_scale_factor + energy_offset


def to_counts(MeV:float, calib_point_1:tuple[float, float]=(0.806, 156745), calib_point_2:tuple[float,float]=(1.679, 320842))->float:
    """
    Converts MeV to ADC counts using the energy calibration points.
    
    The calibration points are defined as tuples of (energy, channel).
    
    Returns the ADC counts.
    """
    energy_1, channel_1 = calib_point_1
    energy_2, channel_2 = calib_point_2
    
    energy_scale_factor = (energy_2 - energy_1) / (channel_2 - channel_1)
    energy_offset = energy_1 - energy_scale_factor * channel_1
    
    return (MeV - energy_offset)/energy_scale_factor