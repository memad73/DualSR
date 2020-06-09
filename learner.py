import torch

class Learner:
    # Default hyper-parameters
    lambda_update_freq = 200
    bic_loss_to_start_change = 0.4
    lambda_bicubic_decay_rate = 100.
    update_l_rate_freq = 750
    update_l_rate_rate = 4.
    lambda_sparse_end = 5
    lambda_centralized_end = 1
    lambda_bicubic_min = 5e-6

    def __init__(self, model):
        self.bic_loss_counter = 0
        self.similar_to_bicubic = False  # Flag indicating when the bicubic similarity is achieved
        self.insert_constraints = True  # Flag is switched to false once constraints are added to the loss
               
        self.G_UP_lr_scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer_G_UP, step_size=750, gamma=0.5)

    def update(self, iteration, model):
        self.G_UP_lr_scheduler.step()
        
        if iteration == 0:
            return
        # Update learning rate every update_l_rate freq
        if iteration % self.update_l_rate_freq == 0:
            for params in model.optimizer_G_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate
            for params in model.optimizer_D_DN.param_groups:
                params['lr'] /= self.update_l_rate_rate

        # Until similar to bicubic is satisfied, don't update any other lambdas
        if not self.similar_to_bicubic:
            if model.regularization.loss_bicubic < self.bic_loss_to_start_change:
                if self.bic_loss_counter >= 2:
                    self.similar_to_bicubic = True
                    #print('similar_to_bicubic, iter=', iteration)
                else:
                    self.bic_loss_counter += 1
            else:
                self.bic_loss_counter = 0
        # Once similar to bicubic is satisfied, consider inserting other constraints
        elif iteration % self.lambda_update_freq == 0 and model.regularization.lambda_bicubic > self.lambda_bicubic_min:
            model.regularization.lambda_bicubic = max(model.regularization.lambda_bicubic / self.lambda_bicubic_decay_rate, self.lambda_bicubic_min)
            if self.insert_constraints and model.regularization.lambda_bicubic < 5e-3:
                model.regularization.lambda_centralized = self.lambda_centralized_end
                model.regularization.lambda_sparse = self.lambda_sparse_end
                self.insert_constraints = False
