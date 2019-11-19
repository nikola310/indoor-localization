
class DeltaSum():

    def __init__(self, delta_x, delta_y, index):
        self._delta_x = delta_x
        self._delta_y = delta_y
        self._index = index

    def __lt__(self, other):
        return self._delta_x < other._delta_x and self._delta_y < other._delta_y

    def __str__(self):
        return " ".join(["Delta x:", str(self._delta_x), ", delta y:", str(self._delta_y), ", index:", str(self._index)])