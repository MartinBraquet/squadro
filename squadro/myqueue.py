class Queue():
  
  def __init__(self):
    self.q = [ ]
    self.index = 0
    self.size = 0
  
  def add(self, value):
    self.q.append(value)
    self.size += 1
  
  def remove(self):
    self.size -= 1
    ret = self.q[self.index]
    self.index += 1
    return ret
  
  def __len__(self):
    return self.size
  
  def __str__(self):
    return str(self.q[self.index:])
  