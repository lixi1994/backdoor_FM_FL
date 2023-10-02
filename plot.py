import pickle

import matplotlib.pyplot as plt
import numpy as np

from options import args_parser

linewidth = 1
markersize = 3
fontsize = 18
desired_fontsize_for_xticks = 20  # Adjust as needed
desired_fontsize_for_yticks = 20  # Adjust as needed
dpi = 200


def plot(args):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    loss_list, acc_list, asr_list = [], [], []
    for mode in ['clean_', 'classicBD_', '']:
        file_name = './save/{}fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_results.pkl'.format(
            mode, args.dataset, args.model, args.epochs, args.frac,
            args.iid, args.local_ep, args.local_bs)
        with open(file_name, 'rb') as f:
            loss, acc, asr = pickle.load(f)
        loss_list.append(loss)
        acc_list.append(acc)
        asr_list.append(asr)

    epochs = np.arange(len(loss_list[0]))

    axs[0].plot(epochs, loss_list[0], '*:', label='clean', linewidth=linewidth, markersize=markersize)
    axs[0].plot(epochs, loss_list[1], 'P--', label='BD_FL', linewidth=linewidth, markersize=markersize)
    axs[0].plot(epochs, loss_list[2], 'o-.', label='BD_FMFL', linewidth=linewidth, markersize=markersize)

    axs[1].plot(epochs, acc_list[0], '*:', label='clean', linewidth=linewidth, markersize=markersize)
    axs[1].plot(epochs, acc_list[1], 'P--', label='BD_FL', linewidth=linewidth, markersize=markersize)
    axs[1].plot(epochs, acc_list[2], 'o-.', label='BD_FMFL', linewidth=linewidth, markersize=markersize)

    axs[2].plot(epochs, asr_list[0], '*:', label='clean', linewidth=linewidth, markersize=markersize)
    axs[2].plot(epochs, asr_list[1], 'P--', label='BD_FL', linewidth=linewidth, markersize=markersize)
    axs[2].plot(epochs, asr_list[2], 'o-.', label='BD_FMFL', linewidth=linewidth, markersize=markersize)

    axs[0].set_ylabel("Average Training Loss", fontsize=fontsize)
    axs[1].set_ylabel("Test Accuracy (ACC)", fontsize=fontsize)
    axs[2].set_ylabel("Attack Success Rate (ASR)", fontsize=fontsize)

    axs[1].set_ylim(0.7, 1.0)
    axs[2].set_ylim(0, 1.1)

    axs[0].legend(loc="best", fontsize=fontsize)

    for a in axs.flat:
        a.set_xlabel("Communication Rounds", fontsize=fontsize)
        # a.legend(loc="best", fontsize=fontsize)
        a.tick_params(axis='x', labelsize=desired_fontsize_for_xticks)
        a.tick_params(axis='y', labelsize=desired_fontsize_for_xticks)
    # fig.text(0.5, -0.04, 'Perturbed frequencies at test-time', ha='center', fontsize=fontsize)
    # fig.text(0.04, 0.5, 'Y-axis', va='center', rotation='vertical')
    # fig.subplots_adjust(bottom=.8)  # Adjust the bottom margin
    # fig.text(0.5, 0.23, 'Perturbed frequencies at test-time', ha='center', fontsize=fontsize)

    # show
    plt.tight_layout()
    plt.show()

    fig.savefig(f'plots/fed_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_'
                f'iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].png', dpi=dpi)



if __name__ == '__main__':
    args = args_parser()
    plot(args)
