from utils import mkdir
import argparse
import os


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the c lass; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--project_id', type=str, default='bf-data-uat-001')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        parser.add_argument('--experiment_name', type=str, default='',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_blob_path', type=str, default='', help='blob of dataset on ml-models bucket')
        parser.add_argument('--download_dataset', action='store_true', help='if download dataset from gcs')
        parser.add_argument('--is_train', action='store_true', help='if train model')
        parser.add_argument('--save', action='store_true', help='if save model')
        parser.add_argument('--deploy', action='store_true', help='if deploy model to ml-models bucket')
        parser.add_argument('--monitoring', action='store_true', help='')
        parser.add_argument('--logger_name', type=str, default='', help='logger name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # dataset parameters
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

        # additional parameters
        parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--seed', type=int, default=1024, help='integer ,to use as random seed.')
        parser.add_argument('--num_folds', type=int, default=5, help='number of folds at k-fold cross validation')
        parser.add_argument('--ndcg', type=str, default='5,10,20', help="set ndcg example '5,10,20'")

        # for setting inputs
        parser.add_argument('--dataset', type=str, default='dataset.pickle', help='dataset file name')
        parser.add_argument('--dataroot', type=str, default='./dataset', help='file path of input dataset')

        # mlflow server parameters
        parser.add_argument('--mlflow_host', type=str, default='http://10.32.192.39:5000')
        parser.add_argument('--mlflow_experiment_id', type=int, default=None)
        parser.add_argument('--mlflow_experiment_run_name', type=str, default=None)

        # mail server setting
        parser.add_argument('--mail_server', type=str, default='smtp.gmail.com')
        parser.add_argument('--mail_server_port', type=str, default='587')
        parser.add_argument('--mail_server_account', type=str, default='bfdataprod002@gmail.com')
        parser.add_argument('--mail_server_password', type=str, default='davjnuouxzwpiqus')

        # online service
        parser.add_argument('--api_model_version_url', type=str, default='http://10.128.33.3:5000/api/model_descript')

        self.initialized = True
        return parser

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""

        # Check if it has been initialized
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # Get the basic options
        opt, _ = parser.parse_known_args()
        self.opt = opt
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print(f'{str(k)}: {str(v)}')
        print('-------------- End ----------------')

        # Create experiment folder
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name)
        mkdir(expr_dir)
        # Create dataset folder
        mkdir(self.opt.dataroot)

        if self.opt.save:
            opt_file_prefix = 'train' if self.opt.is_train else 'test'
            file_name = os.path.join(expr_dir, f'{opt_file_prefix}_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write(f'{str(k)}: {str(v)}\n')
                opt_file.write('-------------- End ----------------\n')

        return self.opt
