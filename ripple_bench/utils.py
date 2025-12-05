import json 

def save_dict(data, savepath):
    with open(savepath, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def read_dict(savepath):
    with open(savepath, 'r') as f:
        data = json.load(f)
    return data
