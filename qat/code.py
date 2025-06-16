from qat import *

######################################################################
# 用户可根据需要修改以下参数
# 1. 模型 ckpt 和编译产出物的保存路径
model_path = "model/mobilenetv2"
# 2. 数据集下载和保存的路径
data_path = "data"
# 3. 训练时使用的 batch_size
train_batch_size = 256
# 4. 预测时使用的 batch_size
eval_batch_size = 256
# 5. 训练的 epoch 数
epoch_num = 30
# 6. 模型保存和执行计算使用的 device
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
######################################################################

train_data_loader, eval_data_loader = prepare_data_loaders(
    data_path, train_batch_size, eval_batch_size
)

class FxQATReadyMobileNetV2(MobileNetV2):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
    ):
        super().__init__(
            num_classes, width_mult, inverted_residual_setting, round_nearest
        )
        self.quant = QuantStub(scale=1 / 128)
        self.dequant = DeQuantStub()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(self.last_channel, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)

##############################float########################################
# 浮点模型初始化
float_model = FxQATReadyMobileNetV2()

# 由于模型的最后一层和预训练模型不一致，需要进行浮点 finetune
optimizer = torch.optim.Adam(
    float_model.parameters(), lr=0.001, weight_decay=1e-3
)
best_acc = 0

for nepoch in range(epoch_num):
    float_model.train()
    train_one_epoch(
        float_model,
        nn.CrossEntropyLoss(),
        optimizer,
        None,
        train_data_loader,
        device,
    )

    # 浮点精度测试
    float_model.eval()
    top1, top5 = evaluate(float_model, eval_data_loader, device)

    print(
        "Float Epoch {}: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
            nepoch, top1.avg, top5.avg
        )
    )

    if top1.avg > best_acc:
        best_acc = top1.avg
        # 保存最佳浮点模型参数
        torch.save(
            float_model.state_dict(),
            os.path.join(model_path, "float-checkpoint.ckpt"),
        )


##############################calib########################################
######################################################################
# 用户可根据需要修改以下参数
# 1. Calibration 时使用的 batch_size
calib_batch_size = 256
# 2. Validation 时使用的 batch_size
eval_batch_size = 256
# 3. Calibration 使用的数据量，配置为 inf 以使用全部数据
num_examples = float("inf")
# 4. 目标硬件平台的代号，此处使用 bayes 举例，实际使用时需要根据实际部署平台做相应替换
march = March.BAYES
######################################################################
# 在进行模型转化前，必须设置好模型将要执行的硬件平台
set_march(march)
# 将模型转化为 Calibration 状态，以统计各处数据的数值分布特征
calib_model = prepare_qat_fx(
    # 输出模型会共享输入模型的 attributes，为不影响 float_model 的后续使用,
    # 此处进行了 deepcopy
    copy.deepcopy(float_model),
    {
        "": default_calib_8bit_fake_quant_qconfig,
        "module_name": {
            # 在模型的输出层为 Conv 或 Linear 时，可以使用 out_qconfig
            # 配置为高精度输出
            "classifier": default_calib_8bit_weight_32bit_out_fake_quant_qconfig,
        },
    },
).to(
    device
)  # prepare_qat_fx 接口不保证输出模型的 device 和输入模型完全一致

# 准备数据集
calib_data_loader, eval_data_loader = prepare_data_loaders(
    data_path, calib_batch_size, eval_batch_size
)

calib_model = prepare_qat_fx(
    # 输出模型会共享输入模型的 attributes，为不影响 float_model 的后续使用,
    # 此处进行了 deepcopy
    copy.deepcopy(float_model),
    {
        "": default_calib_8bit_fake_quant_qconfig,
        "module_name": {
            # 在模型的输出层为 Conv 或 Linear 时，可以使用 out_qconfig
            # 配置为高精度输出
            "classifier": default_calib_8bit_weight_32bit_out_fake_quant_qconfig,
        },
    },
).to(
    device
)  # prepare_qat_fx 接口不保证输出模型的 device 和输入模型完全一致

# 执行 Calibration 过程（不需要 backward）
# 注意此处对模型状态的控制，模型需要处于 eval 状态以使 Bn 的行为符合要求
calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.CALIBRATION)
with torch.no_grad():
    cnt = 0
    for image, target in calib_data_loader:
        image, target = image.to(device), target.to(device)
        calib_model(image)
        print(".", end="", flush=True)
        cnt += image.size(0)
        if cnt >= num_examples:
            break
    print()
# 测试伪量化精度
# 注意此处对模型状态的控制
calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.VALIDATION)
top1, top5 = evaluate(
    calib_model,
    eval_data_loader,
    device,
)
print(
    "Calibration: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
        top1.avg, top5.avg
    )
)
# 保存 Calibration 模型参数
torch.save(
    calib_model.state_dict(),
    os.path.join(model_path, "calib-checkpoint.ckpt"),
)

################################QAT########################################

######################################################################
# 用户可根据需要修改以下参数
# 1. 训练时使用的 batch_size
train_batch_size = 256
# 2. Validation 时使用的 batch_size
eval_batch_size = 256
# 3. 训练的 epoch 数
epoch_num = 3
######################################################################
# 准备数据集
train_data_loader, eval_data_loader = prepare_data_loaders(
    data_path, train_batch_size, eval_batch_size
)

qat_model = prepare_qat_fx(
    copy.deepcopy(float_model),
    {
        "": default_qat_8bit_fake_quant_qconfig,
        "module_name": {
            "classifier": default_qat_8bit_weight_32bit_out_fake_quant_qconfig,
        },
    },
).to(device)
# 加载 Calibration 模型中的量化参数
qat_model.load_state_dict(calib_model.state_dict())

# 作为一个 filetune 过程，量化感知训练一般需要设定较小的学习率
optimizer = torch.optim.Adam(
    qat_model.parameters(), lr=1e-3, weight_decay=1e-4
)
best_acc = 0
for nepoch in range(epoch_num):
    # 注意此处对 QAT 模型 training 状态的控制方法
    qat_model.train()
    set_fake_quantize(qat_model, FakeQuantState.QAT)
    train_one_epoch(
        qat_model,
        nn.CrossEntropyLoss(),
        optimizer,
        None,
        train_data_loader,
        device,
    )
    # 注意此处对 QAT 模型 eval 状态的控制方法
    qat_model.eval()
    set_fake_quantize(qat_model, FakeQuantState.VALIDATION)
    top1, top5 = evaluate(
        qat_model,
        eval_data_loader,
        device,
    )
    print(
        "QAT Epoch {}: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
            nepoch, top1.avg, top5.avg
        )
    )
    if top1.avg > best_acc:
        best_acc = top1.avg
        torch.save(
            qat_model.state_dict(),
            os.path.join(model_path, "qat-checkpoint.ckpt"),
        )

################################Quanti########################################

######################################################################
# 用户可根据需要修改以下参数
# 1. 使用哪个模型作为流程的输入，可以选择 calib_model 或 qat_model
base_model = qat_model
######################################################################

# 将模型转为定点状态
quantized_model = convert_fx(base_model).to(device)

# 测试定点模型精度
top1, top5 = evaluate(
    quantized_model,
    eval_data_loader,
    device,
)
print(
    "Quantized model: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(
        top1.avg, top5.avg
    )
)


from horizon_plugin_pytorch.utils.onnx_helper import export_to_onnx
data = torch.randn(1,3,32,32).to(device)
export_to_onnx(qat_model,data,"qat.onnx")