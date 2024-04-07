import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import logomaker
import pandas as pd
from DNAconversion import one_hot_encode

def quick_loss_plot(data_label_list, loss_type="BCEWithLogitsLoss"):
    '''
    Plot the loss trajectory for each train/test data.

    Parameters:
    - data_label_list (list): A list of tuples containing train_data, test_data, and label.
    - loss_type (str): The type of loss to be plotted. Default is "BCEWithLogitsLoss".

    Returns:
    - None

    This function plots the loss trajectory for each train/test data in the data_label_list.
    It uses matplotlib to create a line plot with train_data plotted with dashed lines and test_data plotted with solid lines.
    The label parameter is used to provide a label for each line in the plot.
    The plot is saved as "loss_plot.png" in the current directory.
    The current figure is then cleared.

    Example usage:
    >>> data_label_list = [(train_data1, test_data1, "Label1"), (train_data2, test_data2, "Label2")]
    >>> quick_loss_plot(data_label_list, loss_type="MSELoss")
    '''

    import matplotlib.pyplot as plt

    plt.clf()
    for i, (train_data, test_data, label) in enumerate(data_label_list):    
        plt.plot(train_data, linestyle='--', color=f"C{i}", label=f"{label} Train")
        plt.plot(test_data, color=f"C{i}", label=f"{label} Val", linewidth=3.0)

    plt.legend()
    plt.ylabel(loss_type)
    plt.xlabel("Epoch")

    plt.legend(bbox_to_anchor=(0.75, 1), loc='upper left')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)

    plt.savefig('loss_plot.png')
    plt.clf()  # clear the current figure

# Get the convolutional layers from the model
def get_conv_layers_from_model(model):
    '''
    Given a trained model, extract its convolutional layers
    
    Parameters:
        model (torch.nn.Module): The trained model from which to extract convolutional layers.
        
    Returns:
        conv_layers (list): A list of the convolutional layers extracted from the model.
        model_weights (list): A list of the weights of the convolutional layers.
        bias_weights (list): A list of the bias weights of the convolutional layers.
    '''
    model_children = list(model.children())
    
    # counter to keep count of the conv layers
    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the actual conv layers in this list
    bias_weights = []
    counter = 0 

    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        # get model type of Conv1d
        if type(model_children[i]) == nn.Conv1d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
            bias_weights.append(model_children[i].bias)

        # also check sequential objects' children for conv1d
        elif type(model_children[i]) == nn.Sequential:
            for child in model_children[i]:
                if type(child) == nn.Conv1d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
                    bias_weights.append(child.bias)

    print(f"Total convolutional layers: {counter}")
    return conv_layers, model_weights, bias_weights

# View the convolutional layers from the model
def view_filters(model_weights, num_cols=8, input_channels=['A','C','G','T']):
    """
    Visualizes the filters of the first convolutional layer in a given model.

    Args:
        model_weights (torch.Tensor): The weights of the model's first convolutional layer.
        num_cols (int, optional): The number of columns in the visualization grid. Defaults to 8.
        input_channels (list, optional): The labels for the input channels. Defaults to ['A','C','G','T'].

    Returns:
        None
    """
    model_weights = model_weights[0]
    num_filt = model_weights.shape[0]
    filt_width = model_weights[0].shape[1]
    num_rows = int(np.ceil(num_filt/num_cols))
    
    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))

    for i, filter in enumerate(model_weights):
        ax = plt.subplot(num_rows, num_cols, i+1)
        ax.imshow(filter.cpu().detach(), cmap='gray')
        ax.set_yticks(np.arange(len(input_channels)))
        ax.set_yticklabels(input_channels)
        ax.set_xticks(np.arange(filt_width))
        ax.set_title(f"Filter {i}")

    plt.tight_layout()
    plt.savefig('conv_filters.png')

def get_conv_output_for_seq(seq, conv_layer, DEVICE, aa=False):
    '''
    Given an input sequence and a convolutional layer, 
    get the output tensor containing the conv filter 
    activations along each position in the sequence

    Args:
        seq (list): The input sequence.
        conv_layer (torch.nn.Module): The convolutional layer.
        DEVICE (constant): The device to run the model on.

    Returns:
        torch.Tensor: The output tensor containing the conv filter activations.
    '''
    # format seq for input to conv layer (OHE, reshape)
    seq = torch.tensor(one_hot_encode(seq, aa=aa)).unsqueeze(0).to(DEVICE)
    # print("Shape of seq:", seq.shape)
    # run seq through conv layer
    with torch.no_grad(): # don't want as part of gradient graph
        # apply learned filters to input seq
        res = conv_layer(seq.float())
        return res[0]
    

def get_filter_activations(seqs, conv_layer, DEVICE, act_thresh=0, input_channels=['A','C','G','T'], aa=False):
    '''
    Given a set of input sequences and a trained convolutional layer, 
    determine the subsequences for which each filter in the conv layer 
    activate most strongly. 
    
    Args:
        seqs (list): A list of input sequences.
        conv_layer (torch.nn.Conv1d): The trained convolutional layer.
        act_thresh (float, optional): The activation threshold. Defaults to 0.
        input_channels (list, optional): The list of input channels. Defaults to ['A','C','G','T'].
    
    Returns:
        dict: A dictionary containing the position weight matrices (PWMs) for each filter in the conv layer.
    
    Algorithm:
        1. Run seq inputs through conv layer.
        2. Loop through filter activations of the resulting tensor, saving the position where filter activations were > act_thresh.
        3. Compile a count matrix for each filter by accumulating subsequences which activate the filter above the threshold act_thresh.
    '''
    # initialize dict of pwms for each filter in the conv layer
    # pwm shape: len(input_channels) X filter width, initialize to 0.0s
    num_filters = conv_layer.out_channels
    filt_width = conv_layer.kernel_size[0]
    filter_pwms = dict((i,torch.zeros(len(input_channels),filt_width)) for i in range(num_filters))
    
    print("Number of filters", num_filters)
    print("Filter width", filt_width)
    
    # loop through a set of sequences and collect subseqs where each filter activated
    for seq in seqs:
        # get a tensor of each conv filter activation along the input seq
        res = get_conv_output_for_seq(seq, conv_layer, DEVICE, aa=aa)

        # for each filter and it's activation vector
        for filt_id, act_vec in enumerate(res):
            # collect the indices where the activation level 
            # was above the threshold
            act_idxs = torch.where(act_vec>act_thresh)[0]
            activated_positions = [x.item() for x in act_idxs]

            # use activated indicies to extract the actual DNA
            # subsequences that caused filter to activate
            for pos in activated_positions:
                subseq = seq[pos:pos+filt_width]
                subseq = subseq.ljust(filt_width, 'X')
                # print("subseq",pos, subseq)
                # transpose OHE to match PWM orientation
                subseq_tensor = torch.tensor(one_hot_encode(subseq, aa=aa))
                # print(subseq_tensor.shape)

                # add this subseq to the pwm count for this filter
                filter_pwms[filt_id] += subseq_tensor            
            
    return filter_pwms

def view_filters_and_logos(model_weights, filter_activations, num_cols=8, input_channels=['A','C','G','T']):
    '''
    Given some convolutional model weights and filter activation PWMs, 
    visualize the heatmap and motif logo pairs in a simple grid.

    Parameters:
    - model_weights (Tensor): The convolutional model weights.
    - filter_activations (List[Tensor]): The filter activation PWMs.
    - num_cols (int, optional): The number of columns in the grid. Default is 8.
    - input_channels (List[str], optional): The input channels. Default is ['A','C','G','T'].

    Returns:
    None
    '''

    model_weights = model_weights[0].squeeze(1)
    # print(model_weights.shape)

    # make sure the model weights agree with the number of filters
    assert(model_weights.shape[0] == len(filter_activations))
    
    num_filts = len(filter_activations)
    num_rows = int(np.ceil(num_filts/num_cols))*2+1 
    # ^ not sure why +1 is needed... complained otherwise
    
    plt.figure(figsize=(20, 17))

    j=0 # use to make sure a filter and its logo end up vertically paired
    for i, filter in enumerate(model_weights):
        if (i)%num_cols == 0:
            j += num_cols

        # display raw filter
        ax1 = plt.subplot(num_rows, num_cols, i+j+1)
        ax1.imshow(filter.cpu().detach(), cmap='gray')
        ax1.set_yticks(np.arange(len(input_channels)))
        ax1.set_yticklabels(input_channels)
        ax1.set_xticks(np.arange(model_weights.shape[2]))
        ax1.set_title(f"Filter {i}")

        # display sequence logo
        ax2 = plt.subplot(num_rows, num_cols, i+j+1+num_cols)
        filt_df = pd.DataFrame(filter_activations[i].T.numpy(),columns=input_channels)
        filt_df_info = logomaker.transform_matrix(filt_df,from_type='counts',to_type='information')
        logomaker.Logo(filt_df_info,ax=ax2)
        ax2.set_title(f"Filter {i}")

    plt.tight_layout()
    plt.savefig('conv_filters_and_logos.png')