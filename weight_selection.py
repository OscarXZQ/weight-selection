import argparse
import numpy as np
import torch
import timm
import os
from models.vision_transformer import *
from models.convnext import *

def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim]-1, s_shape[dim])).long()
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.model_type == 'vit':
        teacher = timm.create_model(args.pretrained_model, pretrained=True)
        student = vit_tiny()
    elif args.model_type == 'convnext':
        teacher = timm.create_model(args.pretrained_model, pretrained=True)
        student = convnext_femto()
    else:
        raise ValueError("Invalid model type specified.")

    teacher_weights = teacher.state_dict()
    student_weights = student.state_dict()

    weight_selection = {}
    for key in student_weights.keys():
        if "head" in key:
            continue
        weight_selection[key] = uniform_element_selection(teacher_weights[key], student_weights[key].shape)

    if args.output_dir.endswith(".pt") or args.output_dir.endswith(".pth"):
        torch.save(weight_selection, os.path.join(args.output_dir))
    else:
        torch.save(weight_selection, os.path.join(args.output_dir, f"{args.model_type}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saved model")
    parser.add_argument("--model_type", type=str, default='vit', choices=['vit', 'convnext'], help="Model type: vit or convnext")
    parser.add_argument("--pretrained_model", type=str, default='vit_small_patch16_224_in21k', help="Pretrained model name for timm.create_model")

    args = parser.parse_args()
    main(args)