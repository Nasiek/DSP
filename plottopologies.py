import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_topologies_output(filename):
    # Initialize a dictionary to store results
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(
                r'Topology: (\w+) \| Epoch: (\d+)/(\d+) \| Batch Size: (\d+) \| LR: ([\d\.e\-]+) \| '
                r'Train Acc: ([\d\.]+) \| Test Acc: ([\d\.]+) \| Train Loss: ([\d\.e\+\-]+) \| Test Loss: ([\d\.e\+\-]+)', line)
            
            if match:
                topo, epoch, no_epochs, bs, lr, train_acc, test_acc, train_loss, test_loss = match.groups()
                epoch = int(epoch)
                no_epochs = int(no_epochs)
                bs = int(bs)
                lr = float(lr)
                train_acc = float(train_acc)
                test_acc = float(test_acc)
                train_loss = float(train_loss)
                test_loss = float(test_loss)
                
                train_diff = abs(train_loss - test_loss)
                acc_diff = abs(train_acc - test_acc)
                
                print(f"Topology: {topo}, Epoch: {epoch}, LR: {lr}, BS: {bs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}")
                print(f"    Loss Difference: {train_diff}, Accuracy Difference: {acc_diff}")
                
                results[topo][(lr, bs)]['train_loss'].append((epoch, train_loss))
                results[topo][(lr, bs)]['test_loss'].append((epoch, test_loss))
                results[topo][(lr, bs)]['train_acc'].append((epoch, train_acc))
                results[topo][(lr, bs)]['test_acc'].append((epoch, test_acc))
    
    return results



def plot_results(results):
    for topo in results:
        for lr_bs in results[topo]:
            lr, bs = lr_bs
            train_loss = results[topo][lr_bs]['train_loss']
            test_loss = results[topo][lr_bs]['test_loss']
            train_acc = results[topo][lr_bs]['train_acc']
            test_acc = results[topo][lr_bs]['test_acc']
            
            epochs_train, losses_train = zip(*train_loss)
            epochs_test, losses_test = zip(*test_loss)
            epochs_train_acc, acc_train = zip(*train_acc)
            epochs_test_acc, acc_test = zip(*test_acc)
            
            loss_diff = [abs(train - test) for train, test in zip(losses_train, losses_test)]
            acc_diff = [abs(train - test) for train, test in zip(acc_train, acc_test)]
            
            # Plot loss difference
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_train, loss_diff, label=f'LR: {lr}, BS: {bs}')
            plt.title(f'Loss Difference ({topo} Topology)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss Difference')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # Plot accuracy difference
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_train_acc, acc_diff, label=f'LR: {lr}, BS: {bs}')
            plt.title(f'Accuracy Difference ({topo} Topology)')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy Difference')
            plt.legend()
            plt.grid(True)
            plt.show()

# def plot_results(results):
#     for topo, lr_bs_dict in results.items():
#         fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
#         plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust space between plots
        
#         # Plot training losses
#         axs[0, 0].set_title(f'Training Loss ({topo})', fontsize=10)
#         axs[0, 0].set_xlabel('Epochs', fontsize=8)
#         axs[0, 0].set_ylabel('Training Loss', fontsize=8)
#         for (lr, bs), metrics in lr_bs_dict.items():
#             epochs, train_losses = zip(*metrics['train_loss'])
#             axs[0, 0].plot(epochs, train_losses, label=f'LR: {lr}, BS: {bs}', linewidth=1)
#         axs[0, 0].legend(fontsize=6)
        
#         # Plot test losses
#         axs[0, 1].set_title(f'Test Loss ({topo})', fontsize=10)
#         axs[0, 1].set_xlabel('Epochs', fontsize=8)
#         axs[0, 1].set_ylabel('Test Loss', fontsize=8)
#         for (lr, bs), metrics in lr_bs_dict.items():
#             epochs, test_losses = zip(*metrics['test_loss'])
#             axs[0, 1].plot(epochs, test_losses, label=f'LR: {lr}, BS: {bs}', linewidth=1)
#         axs[0, 1].legend(fontsize=6)

#         # Plot training accuracy
#         axs[1, 0].set_title(f'Training Accuracy ({topo})', fontsize=10)
#         axs[1, 0].set_xlabel('Epochs', fontsize=8)
#         axs[1, 0].set_ylabel('Training Accuracy', fontsize=8)
#         for (lr, bs), metrics in lr_bs_dict.items():
#             epochs, train_acc = zip(*metrics['train_acc'])
#             axs[1, 0].plot(epochs, train_acc, label=f'LR: {lr}, BS: {bs}', linewidth=1)
#         axs[1, 0].legend(fontsize=6)
        
#         # Plot test accuracy
#         axs[1, 1].set_title(f'Test Accuracy ({topo})', fontsize=10)
#         axs[1, 1].set_xlabel('Epochs', fontsize=8)
#         axs[1, 1].set_ylabel('Test Accuracy', fontsize=8)
#         for (lr, bs), metrics in lr_bs_dict.items():
#             epochs, test_acc = zip(*metrics['test_acc'])
#             axs[1, 1].plot(epochs, test_acc, label=f'LR: {lr}, BS: {bs}', linewidth=1)
#         axs[1, 1].legend(fontsize=6)
        
#         plt.suptitle(f'Results for Topology: {topo}', fontsize=12)
#         plt.show()




def main():
    filename = 'topologiesoutput.txt'
    results = parse_topologies_output(filename)
    plot_results(results)

if __name__ == '__main__':
    main()