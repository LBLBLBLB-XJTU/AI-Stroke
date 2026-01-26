import os
import sys
import argparse

from aistroke.simple_train import simple_train
from aistroke.CV_train import CV_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='aistroke/configs/yamls/trspd.yaml', help='cfg file path')
    parser.add_argument('--gpu', type=str, default='0', help='gpu choosed')
    parser.add_argument("--mode", type=str, default="CV_train", help="running mode")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(PROJECT_ROOT)

    if args.mode == "simple_train":
        simple_train(args, PROJECT_ROOT)
    if args.mode == "CV_train":
        CV_train(args, PROJECT_ROOT)


