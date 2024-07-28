from pathlib import Path

import torch.onnx
import torchvision.models as models
import tyro

import pplcnet

def main(src_model_p: Path, /):
    if "PPLCNet" in src_model_p.name:
        model = pplcnet.PPLCNet_x1_5(num_classes=4)
    elif 'resnet' in src_model_p.name:
        model = models.resnet18(num_classes=4)

    model.load_state_dict(torch.load(src_model_p, map_location="cpu"))    
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    output = model(dummy_input)

    onnx_file_name = src_model_p.with_suffix(".onnx")
    torch.onnx.export(
        model,  # 模型的名称
        dummy_input,  # 一组实例化输入
        onnx_file_name.as_posix(),  # 文件保存路径/名称
        export_params=True,  #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
        #   opset_version=10,          # ONNX 算子集的版本，当前已更新到15
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=["input"],  # 输入模型的张量的名称
        output_names=["output"],  # 输出模型的张量的名称
        # dynamic_axes将batch_size的维度指定为动态，
        # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

if __name__ == "__main__":
    tyro.cli(main)
