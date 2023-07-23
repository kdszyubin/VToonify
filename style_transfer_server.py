from flask import Flask, request, jsonify
import os
import argparse
import numpy as np
import cv2
import dlib
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_video_crop_parameter, tensor2cv2

app = Flask(__name__)

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/077436.jpg', help="path of the content image/video")
        self.parser.add_argument("--style_id", type=int, default=26, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_true", help="transfer the color of the style")
        self.parser.add_argument("--ckpt", type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--scale_image", action="store_true", help="resize and crop the image to best fit the model")
        self.parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path of the extrinsic style code")
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--video", action="store_true", help="if true, video stylization; if false, image stylization")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan | toonify")
        self.parser.add_argument("--padding", type=int, nargs=4, default=[200,200,200,200], help="left, right, top, bottom paddings to the face center")
        self.parser.add_argument("--batch_size", type=int, default=4, help="batch size of frames when processing video")
        self.parser.add_argument("--parsing_map_path", type=str, default=None, help="path of the refined parsing map of the target video")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

parser = TestOptions()
args = parser.parse()
print('*'*98)

options = {
    'cpu': True,
    'ckpt': './checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt',
    'backbone': 'dualstylegan',
    'faceparsing_path': './checkpoint/faceparsing.pth',
    'style_encoder_path': './checkpoint/encoder.pt',
    'exstyle_path': None,
    'style_id': 26,
    'scale_image': True,
    'padding': [600, 600, 600, 600],
    'style_degree': 0.5,
    'color_transfer': False,
}

if args.exstyle_path is None:
    args.exstyle_path = os.path.join(os.path.dirname(args.ckpt), 'exstyle_code.npy')

device = "cpu" if args.cpu else "cuda"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

vtoonify = VToonify(backbone =args.backbone)
vtoonify.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['g_ema'])
vtoonify.to(device)

parsingpredictor = BiSeNet(n_classes=19)
parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
parsingpredictor.to(device).eval()

modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
if not os.path.exists(modelname):
    import wget, bz2
    wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
    zipfile = bz2.BZ2File(modelname+'.bz2')
    data = zipfile.read()
    open(modelname, 'wb').write(data)
landmarkpredictor = dlib.shape_predictor(modelname)

pspencoder = load_psp_standalone(args.style_encoder_path, device)

if args.backbone == 'dualstylegan':
    exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
    stylename = list(exstyles.keys())[args.style_id]
    exstyle = torch.tensor(exstyles[stylename]).to(device)
    with torch.no_grad():
        exstyle = vtoonify.zplus2wplus(exstyle)

print('Load models successfully!')

def process_single_image(input_path, output_path):
    filename = input_path
    basename = os.path.basename(filename).split('.')[0]
    scale = 1
    kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])
    print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])
    if True:
        cropname = os.path.join(output_path, basename + '_input.jpg')

        frame = cv2.imread(filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
        # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
        if args.scale_image:
            paras = get_video_crop_parameter(frame, landmarkpredictor, args.padding)
            if paras is not None:
                h, w, top, bottom, left, right, scale = paras
                H, W = int(bottom - top), int(right - left)
                # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

        with torch.no_grad():
            I = align_face(frame, landmarkpredictor)
            I = transform(I).unsqueeze(dim=0).to(device)
            s_w = pspencoder(I)
            s_w = vtoonify.zplus2wplus(s_w)
            if vtoonify.backbone == 'dualstylegan':
                if args.color_transfer:
                    s_w = exstyle
                else:
                    s_w[:, :7] = exstyle[:, :7]

            x = transform(frame).unsqueeze(dim=0).to(device)
            # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
            # followed by downsampling the parsing maps
            x_p = F.interpolate(
                parsingpredictor(2 * (F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0],
                scale_factor=0.5, recompute_scale_factor=False).detach()
            # we give parsing maps lower weight (1/16)
            inputs = torch.cat((x, x_p / 16.), dim=1)
            # d_s has no effect when backbone is toonify
            y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=args.style_degree)
            y_tilde = torch.clamp(y_tilde, -1, 1)

        cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        save_image(y_tilde[0].cpu(), output_path)
    print('Transfer style successfully!')


@app.route('/vt/process', methods=['POST'])
def process_image():
    # retrieve input_path and output_path from the request
    data = request.get_json()
    input_path = data['input_path']
    output_path = data['output_path']

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    process_single_image(input_path, output_path)

    return jsonify({'output_path': output_path})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
