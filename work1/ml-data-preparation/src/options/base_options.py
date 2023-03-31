from utils import mkdir
import argparse
import os
import uuid
import logging

logging.basicConfig(level=logging.INFO)


class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in data preparation."""
        # basic parameters
        parser.add_argument('--project_id', type=str, default='bf-data-uat-001')
        parser.add_argument('--run_time', type=str, default='')
        parser.add_argument('--content_type', type=str, default=None)
        parser.add_argument('--content_property', type=str, default=None)
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
        parser.add_argument('--experiment_name', type=str, default=None)
        parser.add_argument('--days', type=int, default=30)
        parser.add_argument('--negative_sample_size', type=int, default=20)
        parser.add_argument('--save', action='store_true', default=True, help='if save preparation record')
        parser.add_argument('--upload_gcs', action='store_true', help='if upload experiment to gcs')
        self.initialized = True
        return parser

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.opt = opt
        args = vars(self.opt)

        logging.info(_get_args_info_str(args))
        self.opt.experiment_name = str(uuid.uuid4()).replace('-', '') \
            if self.opt.experiment_name is None else self.opt.experiment_name

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name)
        mkdir(expr_dir)
        if self.opt.save:
            file_name = os.path.join(expr_dir, 'preparation_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(_get_args_info_str(args))
        return self.opt


def _get_args_info_str(args):
    info_str = '------------ Options -------------\n'
    for k, v in sorted(args.items()):
        info_str += f'{str(k)}: {str(v)}\n'
    info_str += '-------------- End ----------------\n'
    return info_str
