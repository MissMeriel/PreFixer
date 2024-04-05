import torch
import sys, os
sys.path.append("C:/Users/Meriel/Documents/GitHub/BeamNGpy/src")
sys.path.append("C:/Users/Meriel/Documents/GitHub/IFAN")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/vqvae/vqvae")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/vae/vqvae")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/vqvae")
sys.path.append("C:/Users/Meriel/Documents/GitHub/supervised-universal-transformation/DAVE2")
from fvcore.nn import FlopCountAnalysis
import torch
from vae.vqvae.vqvae import VQVAE
print(torch.__version__)
# following pytorch recipe
from DAVE2pytorch import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
# device = cpu
def quantize_dyn(model):
    # post training dynamic quantization
    model_dynamic_quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )
    # print(model_dynamic_quantized)
    return model_dynamic_quantized

def quantize_static(model_fp32, input_fp32):
    # print(f"{model_fp32=}")
    # model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # backend = "fbgemm"  # x86 machine
    # torch.backends.quantized.engine = backend
    # model_fp32.qconfig = torch.quantization.get_default_qconfig(backend)
    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    modules_to_fuse = [["conv1", "relu1"], 
                       ["conv2", "relu2"], 
                       ["conv3", "relu3"], 
                       ["lin1", "relu4"],
                       ["lin2", "relu5"],
                       ["lin3", "relu6"],
                    ]
    model_fp32_fused = model_fp32
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32_fused, modules_to_fuse=modules_to_fuse, inplace=True)

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    # input_fp32 = torch.randn(4, 1, 4, 4)
    model_fp32_prepared(input_fp32)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    return model_int8

def calibrate_static_quant_model():
    return


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

'''
To run profiler:
python -m kernprof -l -v .\calc_flops.py
'''

# @profile
def infer(model, input):
    model(input)

def run_benchmark(model, img_loader):
    elapsed = 0
    # model = torch.jit.load(model_file)
    model.eval()
    model.to(device)
    num_batches = 1
    # Run the scripted model on a few batches of images
    for i, hashmap in enumerate(img_loader):
        if i < num_batches:
            images = hashmap['image_base'].float().to(device)
            # images = torch.quantize_per_tensor(images, 0.01, 0, torch.qint8)
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches
    print(f"Elapsed: {elapsed}")
    print('Elapsed time: %3.7f ms' % (elapsed/num_images*1000))
    return elapsed

def main():
    global device, cpu
    transf = "DAVE2"
    if transf == "DAVE2":
        model_name = "../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
        checkpoint = torch.load(model_name, map_location=device).eval()
        model = DAVE2v3(input_shape=(108,192))
        model.load_state_dict(checkpoint.state_dict())
        model.eval()
        model.to(cpu)
        input = torch.zeros((1, 3, 108, 192)).to(cpu)
    elif transf == "resinc":
        input = torch.zeros((1, 3, 270, 480)).to(device)
        model = VQVAE(128, 32, 2, 512, 64, .25, transf="resinc").eval().to(device)
    elif transf == "resdec":
        input = torch.zeros((1,3,54,96)).to(device)
        model = VQVAE(128, 32, 2, 512, 64, .25, transf="resdec").eval().to(device)
        vqvae_name = "../weights/baseline4-50K/portal453969_vqvae50K_resdec_newarch_predloss1.0_bestmodel489.pth"
        checkpoint = torch.load(vqvae_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
    elif transf == "depth":
        input = torch.zeros((1, 3, 108, 192)).to(device)
        model = VQVAE(128, 32, 2, 512, 128, 0.25, transf="depth").eval().to(device)
        vqvae_name = "../weights/baseline4-10K/portal649484_vqvae10K_depth_newenocderarch1_samples10000_predweight1.0_bestmodel490.pth"
        checkpoint = torch.load(vqvae_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
    elif transf == "fisheye":
        input = torch.zeros((1, 3, 108, 192)).to(device)
        model = VQVAE(128, 32, 2, 512, 128, 0.25, transf="fisheye", arch_id=2).eval().to(device)
    print("NORMAL")
    print_size_of_model(model)
    qmodel = quantize_static(model, input)
    # print("quantized", model)
    print("QUANTIZED")
    print_size_of_model(qmodel)

    from UUSTDatasetGenerator import MultiDirectoryDataSequence
    from torch.utils.data import DataLoader
    dataset = MultiDirectoryDataSequence(None, "F:/supervised-transformation-dataset-alltransforms31FULL-V/", image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]),\
                                        robustification=False, noise_level=10, sample_id="STEERING_INPUT",
                                        effect=None) #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    data_loader_test = DataLoader(dataset, batch_size=32, shuffle=True)
    print("NORMAL")
    run_benchmark(model, data_loader_test)
    
    print("QUANTIZED")
    # device=cpu
    run_benchmark(qmodel, data_loader_test)

    exit(0)
    # print("quantized", model.state_dict())
    for i in range(100):
        infer(model, input)
        # qmodel = quantize_dyn(model)
        # qmodel = quantize_static(model, input)
        # infer(qmodel, input)

    exit(0)
    # all static quantization tutorials https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
    # Post-training static quantization involves not just converting the weights from float to int,
    # as in dynamic quantization, but also performing the additional step of first feeding batches of data
    # through the network and computing the resulting distributions of the different activations
    # (specifically, this is done by inserting observer modules at different points that record this data).
    # These distributions are then used to determine how the specifically the different activations should
    # be quantized at inference time (a simple technique would be to simply divide the entire range of activations
    # into 256 levels, but we support more sophisticated methods as well). Importantly, this additional step allows
    # us to pass quantized values between operations instead of converting these values to floats - and then back
    # to ints - between every operation, resulting in a significant speed-up.

    # post training static quantization
    model = torch.load(model_name, map_location=device).eval()
    # backend = "qnnpack"
    # model.qconfig = torch.quantization.get_default_qconfig(backend)
    # torch.backends.quantized.engine = backend
    # model_static_quantized = torch.quantization.prepare(model, inplace=False)
    # model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    num_calibration_batches = 32
    model.eval()

    # Fuse Conv, bn and relu
    model.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model.qconfig = torch.ao.quantization.default_qconfig
    print(model.qconfig)
    torch.ao.quantization.prepare(model, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', model.features[1].conv)

    # Calibrate with the training set
    evaluate(model, criterion, data_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.ao.quantization.convert(model, inplace=True)
    # You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
    # This warning occurs because not all modules are run in each model runs, so some
    # modules may not be calibrated.
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(model)

    top1, top5 = evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

if __name__ == "__main__":
    main()