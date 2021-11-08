import segmentation_models_pytorch as smp
import student_code
import torch

psp = smp.PSPNet(
    encoder_name='resnet50',
    classes=1,
    activation='sigmoid',
)
psp.load_state_dict(torch.load(
    '/Users/AaronLopes/Desktop/cs4476/proj2_release/proj2_code/models/pspnet_resnet50_best_model_weights.pt', map_location=torch.device('cpu')))

print(student_code.print_model_summary(psp))
