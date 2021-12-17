import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "../supervised_only")
import CWRU_extended_task
import CWRU_loader_extended_task
import pandas as pd

rpms = ['1797', '1772', '1750', '1730']
results = pd.DataFrame(columns=rpms)

if __name__ == '__main__':
    for rpm_source in rpms:
        weight_path = '../../models/CWRU/sup_only_final_ext_' + rpm_source + 'rpms.pt'
        f = open("results/CWRU/" + weight_path.split("/")[-1][:-3] + ".txt", "w")
        for rpm_target in rpms:
            results = results.append(pd.DataFrame(index=[rpm_target]))
            model = CWRU_extended_task.Classifier(1000)
            model.load_state_dict(torch.load(weight_path))
            model.eval()
            dataset = CWRU_loader_extended_task.CWRU(1000, rpms=[rpm_target])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            N = 0
            M = 0

            for sample in enumerate(dataloader):
                output = model.forward(sample[1]["data"])
                M += 1
                if torch.argmax(output) == sample[1]["gt"]:
                    N += 1

            f.write('domain shift: ' + rpm_source + '-->' + rpm_target + '\n')
            f.write("Guessed " + str(N) + " of " + str(M) + '. Accuracy: ' + str(round(N / M, 4)) + '\n')
            results[rpm_source][rpm_target] = round(N / M, 4)
        f.close()
    results.to_csv('results/CWRU/' + 'sup_only' + '.csv', ';')
