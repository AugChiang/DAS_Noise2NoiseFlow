class EarlyStopper:
    def __init__(self, patience=1, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self.min_val_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self._counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self._counter += 1
            if self._counter >= self.patience:
                return True
        return False