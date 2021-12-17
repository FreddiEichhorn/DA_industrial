import torch
import sys
from torch.utils.data import DataLoader
sys.path.insert(0, "../supervised_only")
import hydraulic_main_supervised

weight_path = "../../models/Condition monitoring of hydraulic systems/sup_only_2021_11_09_14_12.pt"
model = hydraulic_main_supervised.Classifier()
model.load_state_dict(torch.load(weight_path))
model.eval()


def cond(series):
    return False


dataset = hydraulic_main_supervised.HydraulicTestBench("EPS1.txt", cond, train=False)
loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

if __name__ == '__main__':
    perf_dict = {"cooler": 0, "valve": 0, "pump": 0, "accumulator": 0, "n_samples": 0}

    for sample in enumerate(loader):
        predictions = model.forward(sample[1]["data_100Hz"], sample[1]["data_10Hz"], sample[1]["data_1Hz"])

        for cooler, valve, pump, acc, gt in zip(predictions[0], predictions[1], predictions[2], predictions[3],
                                                sample[1]["gt"]):
            if torch.argmax(cooler) == gt[0]:
                perf_dict["cooler"] += 1
            if torch.argmax(valve) == gt[1]:
                perf_dict["valve"] += 1
            if torch.argmax(pump) == gt[2]:
                perf_dict["pump"] += 1
            if torch.argmax(acc) == gt[3]:
                perf_dict["accumulator"] += 1
            perf_dict["n_samples"] += 1
    f = open("results/" + weight_path.split("/")[-1][:-3] + ".txt", "w")
    f.write(str(perf_dict))
    f.close()
