class NoteDynamic:
  def __init__(self):
    self.absolute = None
    self.relative = []
    self.absolute_position = 0
    self.cresciuto = None


class NoteTempo:
  def __init__(self):
    self.absolute = None
    self.relative = []
    self.time_numerator = 0
    self.time_denominator = 0
    self.recently_changed_position = 0


class NotePedal:
  def __init__(self):
    self.at_start = 0
    self.at_end = 0
    self.refresh = False
    self.refresh_time = 0
    self.cut = False
    self.cut_time = 0
    self.soft = 0
