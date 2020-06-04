from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='.network/results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--create_video', action='store_true', help='create video of output.')
        parser.add_argument('--no_merge', action='store_true', help='Do not merge generated face onto original image')
        parser.add_argument('--raw_to_test', action='store_true', help='Convert raw data to trainable data')
        parser.add_argument('--black_background', action='store_true', help='Adds background of mask to result to see difference more clearly.')
        parser.add_argument('--facenet', action='store_true', help='Calculates facenet distance of result and real.')
        parser.add_argument('--merge_background', action='store_true', help='Merges generated face onto original background.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
