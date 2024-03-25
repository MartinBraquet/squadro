class Stack():

    def __init__(self):
        self.s = [ ]
        self.index = 0
    
    def copy(self):
        s = Stack()
        for x in self:
            s.add(x)
        return s
    
    def clear(self):
      self.index = 0

    def __len__(self):
      return self.index

    def add(self, value):
        if self.index == len(self.s):
            self.s.append(value)
            self.index += 1
        else:
            self.s[self.index] = value
            self.index += 1
    
    def remove(self):
        ret = self.s[self.index - 1]
        self.index -= 1
        return ret

    def top(self):
      return self.s[self.index - 1]
    
    def __iter__(self):
        self.it_index = 0
        return self
    
    def __next__(self):
        if self.it_index == self.index:
            raise StopIteration
        else:
            ret = self.s[self.it_index]
            self.it_index += 1
            return ret
    
    def __str__(self):
        return str(self.s[:self.index])


