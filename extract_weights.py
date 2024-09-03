import torch
import numpy as np

from model import TransformeConfig, TransformerModel


def process_tensor(tensor):
    print(f'Shape: {tuple(tensor.shape)}')

    return tensor.detach().cpu().flatten().tolist()


def extract_weights(checkpoint_path, output_path):
    config = TransformeConfig()
    model = TransformerModel(config)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)


    model.load_state_dict(state_dict)
    
    weights = []

    weights += process_tensor(model.transformer.wte.weight) # wte
    weights += process_tensor(model.transformer.wpe.weight) # wpe
    
    for layer in model.transformer.layers:
        weights += process_tensor(layer.attention.qkv_linear.weight) # w_attn_qkv
        weights += process_tensor(layer.attention.qkv_linear.bias) # b_attn_qkv

        weights += process_tensor(layer.attention.out_proj.weight) # w_attn_out
        weights += process_tensor(layer.attention.out_proj.bias) # b_attn_out
        
        weights += process_tensor(layer.mlp.fc1.weight) # w_dense_0
        weights += process_tensor(layer.mlp.fc1.bias) # b_dense_0
        weights += process_tensor(layer.mlp.fc2.weight) # w_dense_1
        weights += process_tensor(layer.mlp.fc2.bias) # b_dense_1
        
        weights += process_tensor(layer.norm1.weight) + process_tensor(layer.norm2.weight) # ln_gamma
        weights += process_tensor(layer.norm1.bias) + process_tensor(layer.norm2.bias) # ln_beta
    
    weights += process_tensor(model.transformer.lm_head.weight) # w_out
    
    np.array(weights).astype(np.float32).tofile(output_path)


def main():
    checkpoint_path = 'checkpoints/model_epoch_0_step_16384.pt'
    output_path = 'weights.bin'

    extract_weights(checkpoint_path, output_path)


if __name__ == '__main__':
    main()
