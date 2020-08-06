import torch
import torch.nn as nn
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

       
def evaluate(model, loader):
    """Evaluates the trained network on test data."""
    correct = 0
    total = 0
    for data, response in loader:
        predictions = predict(model, data)
        total += predictions.shape[0]
        for i in range(predictions.shape[0]):
            if response[i].item() == predictions[i].item():
                correct += 1
    return correct / total

def predict(model, input_tensor):
    with torch.no_grad():
        model.eval()
        input_matrix = input_tensor.to(device)
        log_probs = model(input_matrix)
        predictions = log_probs.argmax(dim=1)
        return predictions

def train(num_epochs, config, data_loader, multigpu = False):
    
    def maybe_evaluate(model, epoch, prev_best, prev_best_acc):
        best_model = prev_best
        best_test_acc = prev_best_acc
        test_acc = None
        if epoch % 100 == 99:
            test_acc = evaluate(model, test_loader)
            print('epoch {} test: {:.2f}'.format(epoch, test_acc))
            if test_acc > prev_best_acc:
                best_test_acc = test_acc
                best_model = model
                print('saving new model')
                torch.save(best_model, 'best.model2')
        return best_model, best_test_acc, test_acc
    
    def maybe_report_time():
        if False and epoch % 100 == 0 and epoch > 0:
            finish_time = time.clock()
            time_per_epoch = (finish_time - start_time) / epoch
            print('Average time per epoch: {:.2} sec'.format(time_per_epoch))

    start_time = time.clock()
    net_factory = config.create_network_factory()
    model = net_factory(data_loader.input_size(), data_loader.output_size())
    if multigpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)    
    model.to(device)
    loss_function = nn.NLLLoss()
    optimizer = config.create_optimizer_factory()(model.parameters())
    best_model = None
    best_test_acc = -1.0
    scores = []    
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        loader, test_loader = data_loader.get_loaders(epoch)
        for data, response in loader:
            input_matrix = data.to(device)
            log_probs = model(input_matrix)
            loss = loss_function(log_probs, response)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        best_model, best_test_acc, test_acc = maybe_evaluate(model, epoch,
                                                             best_model, 
                                                             best_test_acc)
        if test_acc is not None:
            scores.append((epoch, test_acc))
        if best_test_acc >= .98:
            break
        maybe_report_time()
    return best_model, scores
