# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import code_dinodino.dinodino_vision_transformer as vits
import pickle
import timm
import torch

def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)

def build_dinodino():

    cfg = pickle.load(open('code_dinodino/cfg_dino.pickle', 'rb'))
    student_backbone, _, _ = build_model_from_cfg(cfg)
    model_timm = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
    chunk_dict = {}
    for k, v in model_timm.state_dict().items():
        chunk_dict[k.replace('blocks.', 'blocks.0.')] = v

    chunk_dict.pop('pos_embed')
    student_backbone.load_state_dict(chunk_dict, strict=False)
    print('model only works on cuda!')
    return student_backbone

    # print(student_backbone)

    # student_backbone.to('cuda')

    # x = torch.ones((1,3,400,128)).to('cuda')
    # out = student_backbone.forward(x)
    # print(out)
    # print(out.shape)
