import os
import tqdm

from options import options
from data import create_dataset
from DualSR import DualSR
from learner import Learner


def train_and_eval(conf):
    model = DualSR(conf)
    dataloader = create_dataset(conf)    
    learner = Learner(model)
    
    print('*' * 60 + '\nTraining started ...')
    for iteration, data in enumerate(tqdm.tqdm(dataloader)):
        model.train(data)
        learner.update(iteration, model)
        
    model.eval()


def main():
    opt = options()
    # Run DualSR on all images in the input directory
    for img_name in os.listdir(opt.conf.input_dir):
        conf = opt.get_config(img_name)
        train_and_eval(conf)
    


if __name__ == '__main__':
    main()
