import torch.nn as nn

class LossComputerDINO(nn.Module):
    """
    This loss module calculates the loss between all non-identical pairs of student and 
    teacher outputs. This handles the temperature updates and centering
    """
    
    def __init__(self):
        """ 
        
        """

        # TODO: too high temperature at the start may cause the teacher's training
        # to be unstable. Investigate the use of warm up as necessary

        super().__init__()
        
    def forward(self, student_output, teacher_output):
        """ 
        Calculate the temperature dependent probability distributions and 
        aggregate the losses between all student/teacher views
        """
        pass