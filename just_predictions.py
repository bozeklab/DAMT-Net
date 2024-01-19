import sys

import torch.backends.cudnn as cudnn

import makedatalist as ml
from model.advanced_model import *
from add_arguments import get_arguments
from torch.utils import data
from dataset.target_dataset import targetDataSet_test
from metrics import *
from val import test_model
from utils.postprocessing import *

def test_augmentation2(testmodel, pred_ori, input_size_target, args, usecuda):

    pred_final = pred_ori
    test_aug = 0
    print('the %d test_aug' % test_aug, 'for %s' % args.save_dir)

    testloader = data.DataLoader(
        targetDataSet_test(args.data_dir_test, args.data_list_test,
                          test_aug, crop_size=input_size_target),
        batch_size=1, shuffle=False)

    pred_total, _ = test_model(testmodel, testloader, args.save_dir, test_aug, args.gpu, usecuda,
                                                    test_aug)

    pred_total = postpre(pred_total, args.save_dir, test_aug)
    return pred_final

class Logger(object):
    def __init__(self, filename='logprocess.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def prediction():
    """Create the model and start the training."""
    # start logger
    sys.stdout = Logger(stream=sys.stdout)
    sys.stderr = Logger(stream=sys.stderr)

    usecuda = True
    cudnn.enabled = True
    args = get_arguments()

    # makedatalist
    # TODO przemysl jak dziala
    ml.makedatalist(args.data_dir_test, args.data_list_test)

    if args.model == 'AE':
        model = source2targetNet(in_channels=1, out_channels=2)

    if usecuda:
        cudnn.benchmark = True
        model.cuda(args.gpu)

    model.load_state_dict(torch.load(args.test_model_path, map_location='cuda:0'))
    # model = torch.load(args.restore_from)

    if args.model == 'AE':
        testmodel = model.get_target_segmentation_net()

    pred_ori = np.zeros((94,4096,4096))
    input_size_target = (512, 512)

    pred_final = test_augmentation2(testmodel,pred_ori,input_size_target,args,usecuda=True)


if __name__ == '__main__':
    prediction()