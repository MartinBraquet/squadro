class PrettyDict(dict):
    def __repr__(self):
        return ', '.join([f"{k}={v:.1f}" for k, v in self.items()])
