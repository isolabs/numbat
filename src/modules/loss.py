import torch
import torch.nn as nn
import torch.nn.functional as F

class LossComputerDINO(nn.Module):
    """
    This loss module calculates the loss between all non-identical pairs of student and 
    teacher outputs. This handles the temperature updates and centering
    """
    
    def __init__(self, output_dim):
        """ 
        Call module constructor and set up the centering parameter, which will work 
        on momentum
        """
        super().__init__()

        # We register the centering parameter, which is calculated based on its
        # previous value, and is thus a part of the loss computer's state. We 
        # initialise the centre to be all zero
        # TODO experiment with 1/E - surely all zeroes is a slow uptick
        self.register_buffer("center", torch.zeros(1, output_dim))

    @torch.no_grad()
    def update_center(self, out_teacher, cent_rate_m):
        """
        For every batch, we use a momentum update on the 'center' value
        based on the teacher outputs and a provided momentum to balance the collapse
        of the distillation. We do this calculation with no grads explicitly 
        so as to not waste computation and memory
        """

        # Calculate the current batch center
        n_batch_size = len(out_teacher)
        batch_center = torch.sum(out_teacher, dim=0, keepdim=True) / n_batch_size

        # Perform the EMA update using a momentum rate parameter m > 0
        self.center = cent_rate_m * self.center + (1 - cent_rate_m) * batch_center

    def forward(self, 
                out_student_global, 
                out_student_local, 
                out_teacher,
                temp_student,
                temp_teacher,
                cent_rate_m):
        """ 
        Calculate the temperature dependent probability distributions and 
        aggregate the losses between all student/teacher views
        """

        # student global output shape:      (B * n_global_crops, E)
        # student local output shape:       (B * n_local_crops,  E)
        # teacher (global) output shape:    (B * n_global_crops, E)

        # Recall that every view/crop gets an embedding

        # Note that the temperatures should be selected to cause the probability 
        # distribution to be sharper (more 'confident') in the teacher, 
        # e.g. S=0.1 and T<0.1. The teacher will later have its probability
        # distribution centered, reducing the sharpness of the distribution
        # (a pull towards the uniform distribution)
        
        # In other words, the temperature performs sharpening - it makes the probability
        # distributions collapse towards a single class dominating. This centering
        # operation does the opposite - it makes the probability distribution collapse
        # towards the uniform distribution. A balance of these two prevents collapse

        # First we need to calculate the student and teacher probability
        # distributions. Softmax over the embedding dimension, not the 
        # batch. For the students, we calculate the log of the probability
        # distributions because a single operation is more efficient and 
        # more numerically stable
        log_prob_s_g = F.log_softmax( out_student_global / temp_student, dim=-1 )
        log_prob_s_l = F.log_softmax( out_student_local  / temp_student, dim=-1 )
        log_prob_s   = torch.cat((log_prob_s_g, log_prob_s_l), dim=0) # (B * (n_crops_total), E)
        # Paper says +c, code says -c ?
        prob_t       = F.softmax( (out_teacher - self.center ) / temp_teacher, dim=-1 )

        # We need to calculate the total (scalar) loss and divide it by the number
        # of loss terms to get the average loss per loss term
        total_loss   = 0
        n_loss_terms = 0

        # TODO: this can potentially be vectorised
        for i, prob_t_i in enumerate(prob_t):
            for j, log_prob_s_j in enumerate(log_prob_s):
                # Don't calculate the loss between the same views
                if i == j:
                    continue 

                # Use the implementation of cross entropy to simplify this calc. Note
                # that order is important here - teacher first
                total_loss   += torch.sum( - prob_t_i * log_prob_s_j, dim=-1 ).mean()
                n_loss_terms += 1

        avg_loss_per_view_pair =  total_loss / n_loss_terms
        return avg_loss_per_view_pair