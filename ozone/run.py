import torch
from ozone.train import evaluate
from ozone.experiment import TrainingConfig, BPE_CONFIG
from ozone.oddone import OddOneOutDataset, OddOneOutDataloader


if __name__ == '__main__':
    import sys
    test_file = sys.argv[1]
    num_choice = int(sys.argv[2])
    model = sys.argv[3]
    is_gpu = sys.argv[4]
    if is_gpu == "cpu":
        model = torch.load(model, map_location=torch.device('cpu'))
    else:
        model = torch.load(model)
    config = TrainingConfig(BPE_CONFIG)
    puzzle_gen = config.create_puzzle_generator()
    test_dataset = OddOneOutDataset(puzzle_gen, num_choice, test_file)
    test_dataloader = OddOneOutDataloader(test_dataset).get_loaders()[0]  
    model = model.eval()
    print(evaluate(model, test_dataloader))
