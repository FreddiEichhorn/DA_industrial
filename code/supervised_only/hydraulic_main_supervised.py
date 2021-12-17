import torch
import datetime
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random


class HydraulicTestBench(Dataset):
    """Dataset class for the Condition monitoring of a hydraulic test bench dataset. Allows arbitrarily splitting the
     dataset in a source and a target domain using one of the time-series names (parameter domain feature) and a
     callable that returns true if this instance should belong to source or target domain.

     :param domain_feature: String, Name of the feature that is used to distinguish the instance between source and
     target domain
     :param domain_cond: Callable that given the time series given by domain_feature decides whether this instance
     belongs to target (True) or source (False) domain
     :param train: Boolean. If true, 5/6 of source and 1/8 of the target domain can be seen, if false the remainder is
     returned
     :param seed: Seed for shuffling training data and source data for reproducibility
     """
    def __init__(self, domain_feature=None, domain_cond=None, train=True, seed=42):
        self.base_reg = "../../data/Condition monitoring of hydraulic systems/"
        self.files_100Hz = ["PS1.txt", "PS2.txt", "PS3.txt", "PS5.txt", "PS6.txt", "EPS1.txt"]
        self.files_10Hz = ["FS1.txt", "FS2.txt"]
        self.files_1Hz = ["TS1.txt", "TS2.txt", "TS3.txt", "TS4.txt", "VS1.txt", "SE.txt", "CE.txt", "CP.txt"]

        # Mapping for gt data
        self.dict_map_cooler = {3: 0, 20: 1, 100: 2}
        self.dict_map_valve = {100: 0, 90: 1, 80: 2, 73: 3}
        self.dict_map_pump = {0: 0, 1: 1, 2: 2}
        self.dict_map_accumulator = {130: 0, 115: 1, 100: 2, 90: 3}

        # Normalization parameters for the dataset
        self.avg_100Hz = torch.Tensor([1.609e+02, 1.095e+02, 1.992e+00, 9.842e+00, 9.728e+00, 2.539e+03]).unsqueeze(1)
        self.avg_10Hz = torch.Tensor([5.8345, 10.3046]).unsqueeze(1)
        self.avg_1Hz = torch.Tensor([52.1929, 40.9788, 38.4710, 31.7453, 0.5770, 59.1572, 39.6014, 1.8628]).unsqueeze(1)

        self.std_100Hz = torch.Tensor([14.78, 15.93, 0.3444, 0.1218, 0.1159, 127.3]).unsqueeze(1)
        self.std_10Hz = torch.Tensor([1.8925, 0.0744]).unsqueeze(1)
        self.std_1Hz = torch.Tensor([3.869, .1938, .049, .5357, .028, 10.78, 7.201, .3428]).unsqueeze(1)

        # split model into source and target domains
        self.target_idx = []
        self.source_idx = []
        self.train = train
        j = 0

        if domain_feature is not None:
            for line in open(self.base_reg + domain_feature, "r"):
                if domain_cond([float(i) for i in line.split("\t")]):
                    self.target_idx.append(j)
                else:
                    self.source_idx.append(j)
                j += 1

        else:
            self.source_idx = list(range(2205))

        random.seed(seed)
        random.shuffle(self.source_idx)
        random.seed(seed)
        random.shuffle(self.target_idx)

    def __getitem__(self, item):
        if self.train:
            if item < 5 * len(self.source_idx) / 6:
                item = self.source_idx[item]
            else:
                item = self.target_idx[item - int(5 * len(self.source_idx) / 6)]
        else:
            if item < len(self.source_idx) / 6:
                item = self.source_idx[item + int(5 * len(self.source_idx) / 6)]
            else:
                item = self.target_idx[item - int(len(self.source_idx) / 6) + int(len(self.target_idx) / 8)]

        out_100hz = []
        j = 0
        for series in self.files_100Hz:
            for line in open(self.base_reg + series, "r"):
                if j == item:
                    out_100hz.append([float(i) for i in line.split("\t")])
                    break
                j += 1

        out_10hz = []
        j = 0
        for series in self.files_10Hz:
            for line in open(self.base_reg + series, "r"):
                if j == item:
                    out_10hz.append([float(i) for i in line.split("\t")])
                    break
                j += 1

        out_1hz = []
        j = 0
        for series in self.files_1Hz:
            for line in open(self.base_reg + series, "r"):
                if j == item:
                    out_1hz.append([float(i) for i in line.split("\t")])
                    break
                j += 1

        j = 0
        for line in open(self.base_reg + "profile.txt", "r"):
            if j == item:
                gt_data = [int(i) for i in line.split("\t")]
                break
            j += 1

        gt_data[0] = self.dict_map_cooler[gt_data[0]]
        gt_data[1] = self.dict_map_valve[gt_data[1]]
        gt_data[2] = self.dict_map_pump[gt_data[2]]
        gt_data[3] = self.dict_map_accumulator[gt_data[3]]

        return dict(data_100Hz=(torch.Tensor(out_100hz) - self.avg_100Hz) / self.std_100Hz,
                    data_10Hz=(torch.Tensor(out_10hz) - self.avg_10Hz) / self.std_10Hz,
                    data_1Hz=(torch.Tensor(out_1hz) - self.avg_1Hz) / self.std_1Hz,
                    gt=torch.Tensor(gt_data[:-1]).to(torch.long))

    def __len__(self):
        if self.train:
            return int(len(self.source_idx) / 6 * 5 + len(self.target_idx) / 8)
        else:
            return int(len(self.source_idx) / 6 + len(self.target_idx) * 7 / 8)

    def get_instance_weights(self):
        """Compute and return the weights for each instance so the dataset appears to be balanced to a model trained on
        it. Made to work with torch.utils.data.WeightedRandomSampler and the state the dataset was in on 11/06/2021. If
        the relative number of instances is changed, the weights need to change. Weights are computed so the class with
        the most numerous instances receives weight 1. All other classes then receive weights greater than 1. Because
        this implementation is inefficient, its result has been save to
        models/Condition monitoring of hydraulic systems/dataset_weights.pt"""

        weights = []
        for item in range(self.__len__()):
            sample = self.__getitem__(item)
            if sample["gt"][1] == 0:
                weight = 1
            else:
                weight = 3.125

            if sample["gt"][2] != 0:
                weight += 2.48
            else:
                weight += 1

            if sample["gt"][3] == 0:
                weight *= 1.35
            elif sample["gt"][3] == 1 or sample["gt"][3] == 2:
                weight *= 2.03
            else:
                weight += 1

            weight += 1
            weights.append(weight)
        return weights


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.l1_100Hz = torch.nn.Conv1d(6, 10, kernel_size=15, stride=3)
        self.l1_100Hz_act = torch.nn.ReLU()
        self.l2_100Hz = torch.nn.Conv1d(10, 16, kernel_size=15, stride=3)
        self.l2_100Hz_act = torch.nn.ReLU()
        self.l3_100Hz = torch.nn.Conv1d(16, 26, kernel_size=15, stride=3)
        self.l3_100Hz_act = torch.nn.ReLU()
        self.l4_100Hz = torch.nn.Conv1d(26, 40, kernel_size=15, stride=3)

        self.l1_10Hz = torch.nn.Conv1d(2, 4, kernel_size=4, stride=2)
        self.l1_10Hz_act = torch.nn.ReLU()
        self.l2_10Hz = torch.nn.Conv1d(4, 8, kernel_size=4, stride=2)
        self.l2_10Hz_act = torch.nn.ReLU()
        self.l3_10Hz = torch.nn.Conv1d(8, 12, kernel_size=4, stride=2)
        self.l3_10Hz_act = torch.nn.ReLU()
        self.l4_10Hz = torch.nn.Conv1d(12, 18, kernel_size=4, stride=2)

        self.l1_1Hz = torch.nn.Conv1d(8, 10, kernel_size=3, stride=2)
        self.l1_1Hz_act = torch.nn.ReLU()
        self.l2_1Hz = torch.nn.Conv1d(10, 14, kernel_size=3, stride=2)
        self.l2_1Hz_act = torch.nn.ReLU()
        self.l3_1Hz = torch.nn.Conv1d(14, 24, kernel_size=3, stride=1)
        self.l3_1Hz_act = torch.nn.ReLU()
        self.l4_1Hz = torch.nn.Conv1d(24, 40, kernel_size=3, stride=1)

        self.l5_act = torch.nn.ReLU()
        self.cooler = torch.nn.Linear(3750, 3)
        self.cooler_softmax = torch.nn.Softmax(1)

        self.valve = torch.nn.Linear(3750, 100)
        self.valve_act = torch.nn.ReLU()
        self.valve2 = torch.nn.Linear(100, 4)
        self.valve_softmax = torch.nn.Softmax(1)

        self.pump = torch.nn.Linear(3750, 3)
        self.pump_softmax = torch.nn.Softmax(1)

        self.accumulator = torch.nn.Linear(3750, 100)
        self.accumulator_act = torch.nn.ReLU()
        self.accumulator2 = torch.nn.Linear(100, 4)
        self.accumulator_softmax = torch.nn.Softmax(1)

    def forward(self, input_100Hz, input_10Hz, input_1Hz):
        x1_100Hz = self.l1_100Hz(input_100Hz)
        x1_100Hz_act = self.l1_100Hz_act(x1_100Hz)
        x2_100Hz = self.l2_100Hz(x1_100Hz_act)
        x2_100Hz_act = self.l2_100Hz_act(x2_100Hz)
        x3_100Hz = self.l3_100Hz(x2_100Hz_act)
        x3_100Hz_act = self.l3_100Hz_act(x3_100Hz)
        x4_100Hz = self.l4_100Hz(x3_100Hz_act)
        x5_100Hz = torch.flatten(x4_100Hz, 1)

        x1_10Hz = self.l1_10Hz(input_10Hz)
        x1_10Hz_act = self.l1_10Hz_act(x1_10Hz)
        x2_10Hz = self.l2_10Hz(x1_10Hz_act)
        x2_10Hz_act = self.l2_10Hz_act(x2_10Hz)
        x3_10Hz = self.l3_10Hz(x2_10Hz_act)
        x3_10Hz_act = self.l3_10Hz_act(x3_10Hz)
        x4_10Hz = self.l4_10Hz(x3_10Hz_act)

        x5_10Hz = torch.flatten(x4_10Hz, 1)

        x1_1Hz = self.l1_1Hz(input_1Hz)
        x1_1Hz_act = self.l1_1Hz_act(x1_1Hz)
        x2_1Hz = self.l2_1Hz(x1_1Hz_act)
        x2_1Hz_act = self.l2_1Hz_act(x2_1Hz)
        x3_1Hz = self.l3_1Hz(x2_1Hz_act)
        x3_1Hz_act = self.l3_1Hz_act(x3_1Hz)
        x4_1Hz = self.l4_1Hz(x3_1Hz_act)

        x5_1Hz = torch.flatten(x4_1Hz, 1)

        feature_vector = torch.cat((x5_100Hz, x5_10Hz, x5_1Hz), 1)
        feature_vector_act = self.l5_act(feature_vector)

        cooler_features = self.cooler(feature_vector_act)
        cooler_scores = self.cooler_softmax(cooler_features)

        valve_features = self.valve(feature_vector_act)
        valve_features_act = self.valve_act(valve_features)
        valve_features2 = self.valve2(valve_features_act)
        valve_scores = self.valve_softmax(valve_features2)

        pump_features = self.pump(feature_vector_act)
        pump_scores = self.pump_softmax(pump_features)

        accumulator_features = self.accumulator(feature_vector_act)
        accumulator_act = self.accumulator_act(accumulator_features)
        accumulator_features2 = self.accumulator2(accumulator_act)
        accumulator_scores = self.accumulator_softmax(accumulator_features2)

        return cooler_scores, valve_scores, pump_scores, accumulator_scores


if __name__ == "__main__":

    def cond(series):
        return False


    dataset = HydraulicTestBench("EPS1.txt", cond)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.load("../../data/Condition monitoring of hydraulic systems/dataset_weightsv2.pt"), len(dataset))
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1, sampler=sampler)
    model = Classifier()
    weight_path = "../../models/Condition monitoring of hydraulic systems/sup_only_2021_11_09_05_42.pt"
    #weight_path = None
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))

    # Losses
    loss_cooler = torch.nn.CrossEntropyLoss()
    loss_valve = torch.nn.CrossEntropyLoss()
    loss_pump = torch.nn.CrossEntropyLoss()
    loss_accumulator = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=.003, momentum=0.1)
    N = 0
    epochs = 100

    for _ in range(epochs):
        for sample in enumerate(loader):
            predictions = model.forward(sample[1]["data_100Hz"], sample[1]["data_10Hz"], sample[1]["data_1Hz"])

            # compute loss
            loss = loss_cooler(predictions[0], sample[1]["gt"][:, 0]) + loss_valve(predictions[1], sample[1]["gt"][:, 1]) +\
                   loss_pump(predictions[2], sample[1]["gt"][:, 2]) + loss_accumulator(predictions[3], sample[1]["gt"][:, 3])
            print(loss)
            loss.backward()
            N += 1
            optimizer.step()
            optimizer.zero_grad()
            if N % 1000 == 0:
                torch.save(model.state_dict(), "../../models/Condition monitoring of hydraulic systems/sup_only_" +
                           datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pt")
