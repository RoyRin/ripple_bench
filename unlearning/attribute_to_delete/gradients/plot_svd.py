# plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            

if __name__ == '__main__':
    dirpath = Path("/n/home04/rrinberg/code/unlearning-with-trak/unlearning/gradients/SVDs")
    # get yamls
    yamls = sorted(list(dirpath.glob("*.yaml")))
    print(f"yamls : {yamls}")

    # get yaml without temp in name
    #yamls = [y for y in yamls if "temp" not in y.stem]

    print(f"yamls : {yamls}")
    yaml_file = yamls[0]
    config = read_yaml(yaml_file)
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    test_acc = config["test_acc"][-1]
    # make legend font size very small
    figsize = (20,10)


    plt.rcParams.update({'legend.fontsize': 'x-small'})

    save_name = f"last_round_of_svds__{model_name}__{dataset_name}"

    SVDs = config["SVDs"]
    last_svds = SVDs[-1]
    print(f"plotting 1")


    # SVDs
    plt.figure(figsize=figsize)
    for k,v in last_svds.items():
        plt.plot(v, label=k)
    # put the legend on the right of hte plot
    plt.legend(ncols = 4, loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.legend(ncols = 4, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.title(f"Last round of SVDs | {model_name} | {dataset_name} | final test acc: {test_acc:.2f}")
    # save plot
    # tight layout
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", bbox_inches='tight')
    plt.show()
    print(f"plotting 2")
    
    # renormalized

    plt.figure(figsize=figsize)
    # plot all svds, renormalized
    for k,v in last_svds.items():
        plt.plot(v/np.linalg.norm(v), label=k)
    plt.legend(ncols = 4, loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.legend(ncols = 4, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.title(f"Last rounds of SVDs, renormalized| {model_name} | {dataset_name} | final test acc: {test_acc:.2f}")
    # save plot
    plt.tight_layout()
    plt.savefig(f"{save_name}__renormalized.png", bbox_inches='tight')
