from .test_options import TestOptions


class PipelineOptions(TestOptions):
    """This class includes pipeline options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = TestOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--no_detect', action='store_true', help='Run dlib detection.')
        parser.add_argument('--no_segment', action='store_true', help='Run segmentation model.')
        parser.add_argument('--no_generate', action='store_true', help='Run generation model.')
        parser.add_argument('--no_blend', action='store_true', help='Run blending model.')

        #
        parser.add_argument('--raw_file_type', type=str, default=".jpg", help='use eval mode during test time.')
        #
        parser.add_argument('--smooth_edge', type=int, default=20, help='How much the edges should be smoothed.')
        parser.add_argument('--margin', type=int, default=0, help='If > 0  the rest of the image output image will be '
                                                                  'cut of but a small margin remains.')
        parser.add_argument('--warp', action='store_true', help='Warp masks to get more anonimized results')

        #
        parser.add_argument('--resize', action='store_true', help='create video of output.')
        parser.add_argument('--pix2pix', action='store_true', help='Save output as AB images for pix2pix.')

        return parser
