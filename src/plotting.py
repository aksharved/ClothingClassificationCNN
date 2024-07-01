import matplotlib.pyplot as plt

# Plotting loss vs. epochs
def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
