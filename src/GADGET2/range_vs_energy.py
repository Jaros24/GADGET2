import matplotlib.path


def get_RvE_cut_indexes(self, points):
    """
    points: list of (energy, range) tuples defining a cut in RvE
    Energy is in MeV, range in mm
    """
    path = matplotlib.path.Path(points)
    to_return = []
    index = 0
    while index < len(self.good_events):
        this_point = (self.total_energy_MeV[index], self.len_list[index])
        if path.contains_point(this_point):
            to_return.append(index)
        index += 1
    return to_return