import random
import numpy as np
import torch
import time
from termcolor import colored
from sklearn.model_selection import StratifiedKFold

from util.util_process import load_embeddings, load_fasta_ss8, construct_dataset
from model import Model
from util.util_metric import evaluate, reg_loss, EarlyStopping


seed = 42
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(seed)


def train_test(train_iter, valid_iter, protT5_dim = 2560, max_len = 100):
    net = Model(protT5_dim = protT5_dim, max_len = max_len).to(device)
    print(net, '\n')
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable Parameter: " + str(params) + "\n")
    print(f'Total parameters: {params / 1e6:.2f}M')

    lr =  3e-4
    best_valid_acc = 0
    EPOCH = 100

    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr, weight_decay=1e-4) # 1e-2  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor =0.5, patience=6)
    early_stopping = EarlyStopping(patience=10, delta=0)

    best_valid_performance, best_test_performance = None, None
    best_valid_ROC, best_valid_PRC = None, None
    best_epoch = 0

    for epoch in range(EPOCH):
        train_loss_ls = []
        t0 = time.time()

        net.train()
        for protT5_feat, ss, mask, label in train_iter:
            if device:
                protT5_feat, ss, mask, label = protT5_feat.to(device), ss.to(device), mask.to(device), label.to(device)

            output = net(protT5_feat, ss, mask)
            loss = reg_loss(net, output, label).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_ls.append(loss.item())

        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data, _, train_loss, _, _ = evaluate(train_iter, net, reg_loss)
            valid_performance, valid_roc_data, valid_prc_data, label_real, valid_loss, _, _ = evaluate(valid_iter, net, reg_loss)

        results = f"\nepoch: {epoch + 1}, loss: {train_loss:.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tAUC,\tAUPRC, \tSE,\tSP, \tF1, \tPre, \tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],valid_performance[4],
            valid_performance[5], valid_performance[6], valid_performance[7]) + '\n' + '=' * 60
        print(results)

        print(f'\nEpoch:{epoch + 1}, loss:{train_loss:.5f}, time:{time.time() - t0:.2f}\n',
              f'Train_ACC:{train_performance[0]:.4f}|Train_AUC:{train_performance[1]:.4f}|Train_AUPRC:{train_performance[2]:.4f}|Train_Sensitivity:{train_performance[3]:.4f}|Train_Specificity:{train_performance[4]:.4f}|Train_F1:{train_performance[5]:.4f}|Train_Pre:{train_performance[6]:.4f}|Train_MCC:{train_performance[7]:.4f}|Train_loss:{train_loss:.4f}\n'
              f'Valid_ACC:{valid_performance[0]:.4f}|Valid_AUC:{valid_performance[1]:.4f}|Valid_AUPRC:{valid_performance[2]:.4f}|Valid_Sensitivity:{valid_performance[3]:.4f}|Valid_Specificity:{valid_performance[4]:.4f}|Valid_F1:{valid_performance[5]:.4f}|Valid_Pre:{valid_performance[6]:.4f}|Valid_MCC:{valid_performance[7]:.4f}|Valid_loss:{valid_loss:.4f}\n')

        valid_acc = valid_performance[0]
        scheduler.step(valid_acc)

        valid_acc = valid_performance[0]
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1

            best_valid_performance = valid_performance.copy()
            best_valid_ROC = valid_roc_data.copy()
            best_valid_PRC = valid_prc_data.copy()

            best_valid_results = '\n' + '=' * 16 + colored(' Best Valid Performance. Epoch[{}] ', 'red').format(
                best_epoch) + '=' * 16  + '\n[ACC,\tAUC,\tAUPRC,\tSE,\tSP,\tF1,\tPre,\tMCC]\n' + \
                '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_valid_performance[0], best_valid_performance[1], best_valid_performance[2], best_valid_performance[3], best_valid_performance[4],
                best_valid_performance[5], best_valid_performance[6], best_valid_performance[7]) + '\n' + '=' * 60
            print(best_valid_results)

        valid_metric = valid_acc 
        early_stopping(valid_metric, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return (best_valid_performance, best_valid_ROC, best_valid_PRC, best_epoch)



def K_CV(protT5_train_dict, ss_train_dict, mask_train_dict,labels_train_dict, k = 5, max_len = 100):
    train_protein = list(protT5_train_dict.keys())
    train_list = [protT5_train_dict[pid] for pid in train_protein]
    train_ss_list = [ss_train_dict[pid] for pid in train_protein]
    train_mask_list = [mask_train_dict[pid] for pid in train_protein]
    train_labels_list = [labels_train_dict[pid] for pid in train_protein]

    train_array = np.array(train_list)
    train_ss_array = np.array(train_ss_list)
    train_mask_array = np.array(train_mask_list)
    train_labels_array = np.array(train_labels_list)

    CV_perform = []
    CV_test = []

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for iter_k, (train_index, valid_index) in enumerate(skf.split(train_array, train_labels_array)):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)

        train_feat, valid_feat = train_array[train_index], train_array[valid_index]
        train_ss, valid_ss = train_ss_array[train_index], train_ss_array[valid_index]
        train_mask, valid_mask = train_mask_array[train_index], train_mask_array[valid_index]
        train_labels, valid_labels = train_labels_array[train_index], train_labels_array[valid_index]
        

        train_dict = {f"train_{i}": feat for i, feat in enumerate(train_feat)}
        train_ss_dict = {f"train_{i}": ss_id for i, ss_id in enumerate(train_ss)}
        train_mask_dict = {f"train_{i}": label for i, label in enumerate(train_mask)}
        train_labels_dict = {f"train_{i}": label for i, label in enumerate(train_labels)}
        

        valid_dict = {f"valid_{i}": feat for i, feat in enumerate(valid_feat)}
        valid_ss_dict = {f"valid_{i}": ss_id for i, ss_id in enumerate(valid_ss)}
        valid_mask_dict = {f"valid_{i}": label for i, label in enumerate(valid_mask)}
        valid_labels_dict = {f"valid_{i}": label for i, label in enumerate(valid_labels)}

        train_iter = construct_dataset(train_dict, train_ss_dict, train_mask_dict, train_labels_dict, train=True)
        valid_iter = construct_dataset(valid_dict, valid_ss_dict, valid_mask_dict, valid_labels_dict, train=False)

        print(f'train_iter: {len(train_iter)}')
        print(f'valid_iter: {len(valid_iter)}')

        first_key = list(train_dict.keys())[0]
        protT5_dim = train_dict[first_key].shape[1]

        best_valid_performance, best_valid_ROC, best_valid_PRC, best_epoch = train_test(
            train_iter, valid_iter,  protT5_dim=protT5_dim, max_len=max_len)

        print('Cross-validation: best_performance', best_valid_performance)
        CV_perform.append(best_valid_performance)

    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ', 'red') + '=' * 16 +
          '\n[ACC,\tBACC,\tAUC,\tAUPRC,\tSE,\tSP,\tF1,\tPre,\tMCC]\n')

    for k_idx, out in enumerate(CV_perform):
        print('第{}折: {:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            k_idx + 1, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]))

    mean_out = np.array(CV_perform).mean(axis=0)
    std_out = np.array(CV_perform).std(axis=0)

    print('\n' + '=' * 16 + "Mean ± Std" + '=' * 16)
    print(
        'ACC:{:.4f}±{:.4f}, \tAUC: {:.4f}±{:.4f},\tAUPRC: {:.4f}±{:.4f},\tSE: {:.4f}±{:.4f},\tSP: {:.4f}±{:.4f},\tF1: {:.4f}±{:.4f},\tPre: {:.4f}±{:.4f},\tMCC: {:.4f}±{:.4f}'.format(
            mean_out[0], std_out[0], mean_out[1], std_out[1], mean_out[2], std_out[2], mean_out[3], std_out[3], 
            mean_out[4], std_out[4], mean_out[5], std_out[5], mean_out[6], std_out[6], mean_out[7], std_out[7]))

    print('\n' + '=' * 60)
    return CV_perform, CV_test


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    max_len = 100
    protT5_folder = 'datasets/NeuroP_feature/ProtT5'
    protT5_train_dict, labels_train_dict = load_embeddings(protT5_folder, max_length=max_len, is_training=True)
    protT5_test_dict, labels_test_dict = load_embeddings(protT5_folder, max_length=max_len, is_training=False)

    ss_folder = 'datasets/NeuroP_feature/ss8'
    ss_train_dict, _ , mask_train_dict = load_fasta_ss8(ss_folder, max_length=max_len, is_training=True)
    ss_test_dict, _ , mask_test_dict = load_fasta_ss8(ss_folder, max_length=max_len, is_training=False)

    first_key = list(protT5_train_dict.keys())[0]
    protT5_dim = protT5_train_dict[first_key].shape[1]
    print(f"ProtT5 feature dimension: {protT5_dim}")

    k_fold = 10
    print(f"Starting {k_fold}-fold cross validation")
    CV_perform, CV_test = K_CV(protT5_train_dict, ss_train_dict, mask_train_dict, labels_train_dict,
                               k=k_fold, max_len = max_len)