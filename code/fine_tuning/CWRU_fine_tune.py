import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("../supervised_only")
import CWRU_loader
import datetime


class Classifier(torch.nn.Module):
    def __init__(self, sample_length):
        super(Classifier, self).__init__()
        self.l1 = torch.nn.Conv1d(3, 16, 3)
        self.l1_act = torch.nn.ReLU()
        self.l1_dropout = torch.nn.Dropout()
        self.l2 = torch.nn.Conv1d(16, 1, 3)
        self.l2_act = torch.nn.ReLU()
        self.l2_dropout = torch.nn.Dropout()
        self.l3 = torch.nn.Linear(sample_length-4, 256)
        self.l3_act = torch.nn.ReLU()
        self.l4 = torch.nn.Linear(256, 128)
        self.l4_act = torch.nn.ReLU()
        self.l5 = torch.nn.Linear(128, 3)
        self.out = torch.nn.Softmax(2)

    def forward(self, input):
        x1 = self.l1(input)
        x1_act = self.l1_act(x1)
        x1_dropout = self.l1_dropout(x1_act)
        x2 = self.l2(x1_dropout)
        x2_act = self.l2_act(x2)
        x2_dropout = self.l2_dropout(x2_act)
        x3 = self.l3(x2_dropout)
        x3_act = self.l3_act(x3)
        x4 = self.l4(x3_act)
        x4_act = self.l4_act(x4)
        x5 = self.l5(x4_act)
        output = self.out(x5)
        return output.squeeze(1)


if __name__ == "__main__":
    sample_length = 1000
    dataset = CWRU_loader.CWRU(sample_length, rpm=1795)

    # sampler = torch.utils.data.WeightedRandomSampler([1.0] * 1211 + [3.5] * 364 + [3.5] * 364, len(dataset))
    sampler = torch.utils.data.WeightedRandomSampler([1.0] * 243 + [2.0] * 121 + [2.0] * 122, len(dataset))
    dataloader = DataLoader(dataset, batch_size=250, shuffle=False, num_workers=1, sampler=sampler)
    model = Classifier(sample_length)
    model.train()
    # weight_path = "../../models/CWRU/fine_tune_2021_11_08_19_02.pt"
    weight_path = None
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.0001)
    N = 0
    num_epochs = 400

    # First learn a model on the full source domain
    for _ in range(num_epochs):
        for sample in enumerate(dataloader):
            output = model.forward(sample[1]["data"])
            loss = loss_function(output, sample[1]["gt"])
            loss.backward()
            print(loss)
            optimizer.step()
            N += 1
            if N % 2000 == 0:
                torch.save(model.state_dict(), "../../models/CWRU/fine_tune_" +
                           datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pt")

    # Pull 10 samples from target data and fine tune on them
    target_set = CWRU_loader.CWRU(sample_length, rpm=1731, train=False)
    target_idx = [0, 1, 2, 485, 486, 487, 488, 608, 609, 610]
    target_data = []
    print("_________")
    print("fine tuning")
    for i in target_idx:
        target_data.append(target_set[i])

    # Train for 50 epochs (should ensure convergence)
    for epoch in range(50):
        for sample in target_data:
            output = model.forward(sample["data"].unsqueeze(0))
            loss = loss_function(output, torch.Tensor([sample["gt"]]).type(torch.LongTensor))
            loss.backward()
            print(loss)
            optimizer.step()
            if epoch % 10 == 0:
                torch.save(model.state_dict(), "../../models/CWRU/fine_tune_" + str(epoch) + "epochs_" +
                           datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pt")

    torch.save(model.state_dict(), "../../models/CWRU/fine_tune_final_" +
               datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pt")
