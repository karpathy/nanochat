class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
        # Cache the most recently aggregated sums for logging/debugging.
        # These values persist across reset_* calls.
        self.last_aux_loss_sum = 0.0
        self.last_router_z_loss_sum = 0.0
    
    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []
    
    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)
    
    def aggregate_aux_loss(self):
        s = sum(self.aux_loss)
        self.last_aux_loss_sum = s 
        return s

    def aggregate_router_z_loss(self):
        s = sum(self.router_z_loss)
        self.last_router_z_loss_sum = s 
        return s

MANAGER = MOEManager()

