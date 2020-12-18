import pandas as pd
import numpy as np
import csv

def result_analyse():
    labels = ['CVSS2_Conf', 'CVSS2_Integrity', 'CVSS2_Avail', 'CVSS2_AccessVect',
              'CVSS2_AccessComp', 'CVSS2_Auth', 'CVSS2_Severity']
    txt_filepath = "/home/zhaodaxing/MSR2019---NN/val_with_time_layer6.txt"
    f1_csv_filepath = '/home/zhaodaxing/MSR2019---NN/layer6_f1_result.csv'
    acc_csv_filepath = '/home/zhaodaxing/MSR2019---NN/layer6_acc_result.csv'

    # 写入F1的结果
    with open(f1_csv_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(labels)
        for i in range(7):
            data = []
            for j in range(1, 9):
                with open(txt_filepath, 'r') as file:
                    total_acc = []
                    total_f1 = []
                    while True:
                        line1 = file.readline().rstrip()
                        line2 = file.readline()
                        line3 = file.readline()
                        line4 = file.readline()
                        line5 = file.readline().rstrip()
                        line6 = file.readline().rstrip()
                        if not line1:
                            break
                        label = line1[13:-10]
                        # year = int(line1[-5:-1])
                        config = int(line1[6])
                        accuracy = float(line6.split(" ")[1])
                        F1_list = [float(line5.split(" ")[1]), float(line5.split(" ")[2]), float(line5.split(" ")[3])]
                        F1 = (F1_list[0] + F1_list[1] + F1_list[2]) / 3

                        if config == j and label == labels[i]:
                            total_f1.append(F1)
                            total_acc.append(accuracy)
                    data.append(sum(total_f1)/500)
            # print(data)
            writer.writerow(data)
                    # print(sum(total_acc)/500)
                    # print(sum(total_f1)/500)

    # 写入accuracy的结果
    with open(acc_csv_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(labels)
        for i in range(7):
            data = []
            for j in range(1, 9):
                with open(txt_filepath, 'r') as file:
                    total_acc = []
                    total_f1 = []
                    while True:
                        line1 = file.readline().rstrip()
                        line2 = file.readline()
                        line3 = file.readline()
                        line4 = file.readline()
                        line5 = file.readline().rstrip()
                        line6 = file.readline().rstrip()
                        if not line1:
                            break
                        label = line1[13:-10]
                        # year = int(line1[-5:-1])
                        config = int(line1[6])
                        accuracy = float(line6.split(" ")[1])
                        F1_list = [float(line5.split(" ")[1]), float(line5.split(" ")[2]), float(line5.split(" ")[3])]
                        F1 = (F1_list[0] + F1_list[1] + F1_list[2]) / 3

                        if config == j and label == labels[i]:
                            total_f1.append(F1)
                            total_acc.append(accuracy)
                    data.append(sum(total_acc)/500)
            # print(data)
            writer.writerow(data)


if __name__ == '__main__':
    result_analyse()