import torch
import hydraulic_main_supervised as hms
from torch.utils.data import DataLoader

def cond(benis):
    return False


if __name__ == '__main__':
    dataset = hms.HydraulicTestBench("EPS1.txt", cond)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.load("../../data/Condition monitoring of hydraulic systems/dataset_weightsv2.pt"), len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, sampler=sampler)

    cooler_stats = [0, 0, 0]
    valve_stats = [0, 0, 0, 0]
    pump_stats = [0, 0, 0]
    acc_stats = [0, 0, 0, 0]

    for sample in enumerate(loader):
        cooler_stats[sample[1]["gt"][0, 0]] += 1
        valve_stats[sample[1]["gt"][0, 1]] += 1
        pump_stats[sample[1]["gt"][0, 2]] += 1
        acc_stats[sample[1]["gt"][0, 3]] += 1
    print("cooler: ", cooler_stats)
    print("valve: ", valve_stats)
    print("pump: ", pump_stats)
    print("acc: ", acc_stats)
