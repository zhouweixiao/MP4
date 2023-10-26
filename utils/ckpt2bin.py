import torch
import argparse
from collections import OrderedDict


def remove_model_prefixes(od: OrderedDict):
    new_od = OrderedDict()
    for key, value in od.items():
        if key.startswith("model.model."):
            new_key = key.replace("model.model.", "", 1)
        elif key.startswith("model."):
            new_key = key.replace("model.", "", 1)
        else:
            new_key = key
        new_od[new_key] = value
    return new_od


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt_path", type=str, required=True, help="The path to the ckpt model weights.")
    parser.add_argument("--bin_path", type=str, required=True, help="The path to the bin model weights.")

    args = parser.parse_args()
        
    checkpoint = remove_model_prefixes(torch.load(args.ckpt_path, map_location='cpu')['state_dict'])
    del checkpoint['lm_head.weight']
    del checkpoint['final_logits_bias']
    
    torch.save(checkpoint, args.bin_path)
    print("The model weights transformation is complete.")
