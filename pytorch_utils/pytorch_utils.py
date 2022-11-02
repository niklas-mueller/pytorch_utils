import torch, torchvision
import torch.nn as nn
import numpy as np
import time
from matplotlib.backends.backend_pdf import PdfPages # This is stupid but needs to be done in order to save as pdf pages
from matplotlib import pyplot as plt

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    fig = plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    return fig

def visualize_layers(model):
    figs = []
    _model = model.module if type(model) == nn.DataParallel else model
    for name, module in _model.named_modules():
        if not isinstance(module, nn.Sequential):
            if type(module) == nn.modules.conv.Conv2d or type(module) == nn.Conv2d:
                filter = module.weight.cpu().data.clone()
            else:
                continue
            fig = visTensor(filter, ch=0, allkernels=True)
            figs.append(fig)
            plt.axis('off')
            plt.title(f'Layer: {name}')
            plt.ioff()

    return figs

def evaluate(loader, model:nn.DataParallel, criterion, device=None, verbose:bool=False):
    num_correct = 0
    num_samples = 0
    losses = {}
    model.eval()

    if device is None:
        if hasattr(model, 'src_device_obj'):
            device = model.src_device_obj
        elif hasattr(model, 'device'):
            device = model.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            loss = criterion(scores, y)
            losses[i] = loss.cpu().tolist()
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        accuracy = float(num_correct)/float(num_samples)
        if verbose:
            print(f'Got {num_correct} \t/ {num_samples} correct -> accuracy {accuracy*100:.2f} %') 
    
    model.train()

    return {'accuracy': accuracy, 'batch_losses': losses}


def train(model, trainloader, valloader, loss_fn, optimizer, device, 
            n_epochs:int, result_manager=None, verbose:bool=True,
            eval_valid_every:int=0, testloader=None, visualize_layers:bool=False):
    

    current_time = time.ctime().replace(' ', '_')
    results = {}
    model.train(True)
    losses = []
    epoch_times = []

    best_valid_loss = np.inf
    best_training_loss = np.inf
    best_model_state_dict = None
    for epoch in range(n_epochs):
        if verbose:
            print(f"Running epoch {epoch}")

        start_time_epoch = time.time()
        # Initalize losses
        # running_loss = 0.0
        # last_loss = 0.0

        # Training loop
        for train_index, (inputs, labels) in enumerate(trainloader):

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # # Gather data and report
            # running_loss += loss.item()

            if eval_valid_every > 0 and train_index % eval_valid_every == eval_valid_every-1:
                # last_loss = running_loss / eval_valid_every # Loss per batch
                # running_loss = 0.0

                eval = evaluate(loader=valloader, model=model, criterion=loss_fn, device=device, verbose=False)
                avg_val_loss = np.mean([batch_losses for _, batch_losses in eval['batch_losses'].items()])
                if avg_val_loss < best_valid_loss:
                    best_valid_loss = avg_val_loss
                    results[f'validation_during_training_epoch-{epoch}'] = eval

        end_time_epoch = time.time()
        epoch_times.append(end_time_epoch - start_time_epoch)
        
        losses.append(loss.item())
        
        if loss.item() < best_training_loss:
            best_training_loss = loss.item()
            if verbose:
                print(f"Found new best model: Saving model in epoch {epoch} with loss {loss.item()}.")
            if result_manager is not None:
                result_manager.save_model(model, filename=f'best_model_{current_time}.pth', overwrite=True)

            best_model_state_dict = model.state_dict()
        if verbose:
            print(f"Loss after epoch {epoch}: {loss.item()}")

    model.train(False)
    
    
    results['training_losses'] = losses
    results['epoch_times'] = epoch_times

    if verbose:
        print(f'Finished Training with loss: {loss.item()}')
        print(f'Average time per epoch for {n_epochs} epochs: {np.mean(epoch_times)}')


    if testloader is not None:
        # Load best model
        model.load_state_dict(best_model_state_dict)

        eval = evaluate(loader=testloader, model=model, criterion=loss_fn, verbose=True)
        results['eval_trained_testdata'] = eval

        if verbose:
            print(f"Evaluated test data: Accuracy: {eval['accuracy']}")

    if result_manager is not None:
        result_manager.save_result(results, filename=f'training_results_{current_time}.yml')
        result_manager.save_model(model, filename=f'final_model_{current_time}.pth')

        if verbose:
            print(f"Save results and model state.")

    if visualize_layers:
        figs = visualize_layers(model)
        if result_manager is not None:
            print("To save the visualisation provide a result manager!")
            result_manager.save_pdf(figs=figs, filename='layer_visualisation_after_training.pdf')

        if verbose:
            print(f"Saved visualization of layers after training.")

    if verbose:
        print(f"Done!")