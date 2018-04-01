import matplotlib.pyplot as plt
import numpy as np
 
def draw_training(loss_curve):
    plt.figure(1)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")  

    plt.plot(loss_curve)
    plt.show()

    
    

