import torch.nn as nn

class LossDINO(nn.Module):
    """
    This loss module calculates the loss between all non-identical pairs of student and 
    teacher outputs. This handles the temperature updates and centering
    """
    
    def __init__(self):
        """ 
        
        """
        super().__init__()
        
    def forward(self, student_output, teacher_output):
        """ 
        Calculate the temperature consistent probability distributions and 
        aggregate the losses between all student/teacher views
        """
        pass