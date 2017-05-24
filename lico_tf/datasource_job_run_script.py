import os
import sys
import shutil
import tarfile
import zipfile
import json
import logging
import logging.config
from os.path import join as path_join
# import rarfile
from PIL import Image
from PIL import ImageOps

args_file = sys.argv[1]
with open(args_file, 'r') as f:
    args = f.readline()
data_source = json.loads(args)
temp_data = data_source['file_name'].split('.')
log_file = path_join(data_source['location'], temp_data[0] + '_preprocess.log')

# ----------------------config log------------------------#
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s.%(funcName)s Line:%(lineno)d %(processName)s %(threadName)s %(message)s'
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        },
    },
    'handlers': {

        'file': {
            'level': "DEBUG",
            'class': "logging.FileHandler",
            'filename': log_file,
            'formatter': 'verbose',
            # 'maxBytes': 20 * 1024 * 1024,
            # 'backupCount': 10
        }
    },
    'loggers': {
        '': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
            'formatter': 'verbose'
        },
    }
}

logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

# ----------------------image_pretreate code------------------------#
# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
ROTATE_90 = 2
ROTATE_180 = 3
ROTATE_270 = 4

# resampling filters
NONE = 0
NEAREST = 0
ANTIALIAS = 1  # 3-lobed lanczos
LINEAR = BILINEAR = 2
CUBIC = BICUBIC = 3


class ImageProcess(object):
    def get_image_operator(self, image_path):
        """
        :param image_path:
        :return: image operator
        """
        try:
            im = Image.open(image_path)
            return im
        except IOError as exc:
            raise exc

    def save_image(self, image_path, dir, im):
        """
        :image_path: the source image path
        :param dir: the directory to save iamge
        :im: image operator
        :return:
        """
        try:
            name = os.path.basename(image_path)
            im.save(dir + '/' + name)
            return
        except Exception as exc:
            raise exc

    def color_or_grayscale(self, im, type):
        """
        Convert the image to grayscale or color.

        :param im: image operator
        :return: image operator
        """
        try:
            if type == 'Color':
                # todo maohaijun, need check args
                im = ImageOps.colorize(im, black=(255, 0, 0), white=(0, 255, 0))
            else:
                im = im.convert("L")
            return im
        except Exception:
            # do nothing
            pass

    def rotate(self, im, angle, resample=NEAREST, expand=0):
        """
        :param im: image operator
        :param angle: In degrees counter clockwise.
        :param resample: An optional resampling filter.  This can be
               one of :py:attr:`PIL.Image.NEAREST` (use nearest neighbour),
               :py:attr:`PIL.Image.BILINEAR` (linear interpolation in a 2x2
               environment), or :py:attr:`PIL.Image.BICUBIC`
               (cubic spline interpolation in a 4x4 environment).
               If omitted, or if the image has mode "1" or "P", it is
               set :py:attr:`PIL.Image.NEAREST`.
        :param expand: Optional expansion flag.  If true, expands the output
               image to make it large enough to hold the entire rotated image.
               If false or omitted, make the output image the same size as the
               input image.
        :return: image operator
        """

        try:
            im = im.rotate(angle, resample=resample, expand=expand)
            return im
        except Exception as exc:
            raise exc

    def shift(self, im, width, height, fill=0):
        """
        :param im: image operator
        :param fill: Pixel fill value (a color value).  Default is 0 (black).
        :return: image operator
        """

        try:
            w, h = im.size
            im = ImageOps.expand(im,
                                 border=(width, height, width + w,
                                         height + h), fill=fill)
            im = im.crop(box=(0, 0, w, h))
            return im
        except Exception as exc:
            raise exc

    def zoom(self, im, width, height, resample=NEAREST):
        """
        :param im: image operator
        :param resample: An optional resampling filter.  This can be
               one of :py:attr:`PIL.Image.NEAREST` (use nearest neighbour),
               :py:attr:`PIL.Image.BILINEAR` (linear interpolation in a 2x2
               environment), :py:attr:`PIL.Image.BICUBIC` (cubic spline
               interpolation in a 4x4 environment), or
               :py:attr:`PIL.Image.ANTIALIAS` (a high-quality downsampling filter).
               If omitted, or if the image has mode "1" or "P", it is
               set :py:attr:`PIL.Image.NEAREST`.
        :return: image operator
        """

        try:
            im = im.resize((width, height), resample=resample)
            return im
        except Exception as exc:
            raise exc

    def mirror(self, im, method=FLIP_LEFT_RIGHT):
        """
        :param im: image operator
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
              :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
              :py:attr:`PIL.Image.ROTATE_180`, or :py:attr:`PIL.Image.ROTATE_270`.
        :return: image operator
        """

        try:
            im = im.transpose(method)
            return im
        except Exception as exc:
            raise exc

    def squash(self, im, width, height, resample=NEAREST):
        """
        :param im: image operator
        :param resample: Optional resampling filter.  This can be one
               of :py:attr:`PIL.Image.NEAREST`, :py:attr:`PIL.Image.BILINEAR`,
               :py:attr:`PIL.Image.BICUBIC`, or :py:attr:`PIL.Image.ANTIALIAS`
               (best quality).  If omitted, it defaults to
               :py:attr:`PIL.Image.NEAREST` (this will be changed to ANTIALIAS in a
               future version).
        :return: image operator
        """

        return self.zoom(im, width, height, resample=resample)

    def crop(self, im, width, height):
        """
        :param im: image operator
        :return: image operator
        """

        try:
            im = im.crop((0, 0, width, height))
            return im
        except Exception as exc:
            raise exc

    def fill(self, im, width, height, fill=0):
        """
        :param im: image operator
        :param fill: Pixel fill value (a color value).  Default is 0 (black).
        :return: image operator
        """

        try:
            w, h = im.size
            im = ImageOps.expand(im, border=((width - w) / 2, (height - h) / 2,
                                             (width - w) / 2, (height - h) / 2), fill=fill)
            return im
        except Exception as exc:
            raise exc

    def half_fill_half_crop(self, im, width, height, resample=NEAREST):
        """
        Do resize to expand the image, then crop it.

        :param im: image operator
        :return: image operator
        """

        try:
            w, h = im.size
            im = im.resize(((w + width) / 2, (h + height) / 2), resample=resample)
            im = im.crop(box=(0, 0, width, height))
            return im
        except Exception as exc:
            raise exc


# ----------------------datasource job code------------------------#
def print_msg(msg):
    print msg
    logger.debug(msg)


class RunImageProcessJob(object):
    def __init__(self, **data_source):
        # decompress
        tar_file = path_join(data_source['tar_path'], data_source['file_name'])
        print_msg('Decompress the file to dir %s......' % data_source['location'])
        datasource_dir = self.__decompress_image_package(tar_file, data_source['location'])
        print_msg("Decompress done!")

        image_save_dir = "%s_preprocess" % datasource_dir
        os.mkdir(image_save_dir)
        print_msg('Image preprocess in dir %s......' % image_save_dir)
        self.__image_pretreat(datasource_dir, image_save_dir, **data_source)
        print_msg("Image preprocess done!")

        # image classify
        print_msg("Image classify begin......")
        self.__classify_data(image_save_dir, data_source['data_distribution_train'],
                             data_source['data_distribution_valid'],
                             data_source['data_distribution_test'])
        print_msg("Classify done!")

    def __decompress_image_package(self, tar_file, dest_dir):
        data = os.path.basename(tar_file).split('.')
        save_dir = path_join(dest_dir, data[0])

        if data[-1] in ['tar', 'gz']:
            with tarfile.open(tar_file) as tar:
                tar.extractall(path=dest_dir)
        elif data[-1] == 'zip':
            with zipfile.ZipFile(tar_file) as zip:
                for name in zip.namelist():
                    zip.extract(name, path=dest_dir)
        # elif data[-1] == 'rar':
        #     with rarfile.RarFile(tar_file) as rar:
        #         rar.extract(path=dest_dir)

        return save_dir

    def __image_pretreat(self, source_dir, dest_dir, **data_source):
        for dirname in os.listdir(source_dir):
            dir_path = path_join(source_dir, dirname)
            if os.path.isdir(dir_path):
                temp_dir = path_join(source_dir, dirname)
                sub_save_dir = path_join(dest_dir, dirname)
                os.mkdir(sub_save_dir)
                for parent, dirnames, filenames in os.walk(temp_dir):
                    length = len(filenames)
                    count = 1
                    for filename in filenames:
                        print_msg("Preprocessing %d / %d in dir: %s, image name: %s" %
                                  (count, length, temp_dir, filename))
                        image_path = path_join(temp_dir, filename)
                        self.__do_pretreat(image_path, sub_save_dir, **data_source)
                        count += 1

    def __do_pretreat(self, image_path, sub_save_dir, **data_source):
        image_ps = ImageProcess()
        im = image_ps.get_image_operator(image_path)
        # color or grayscale
        im = image_ps.color_or_grayscale(im, data_source['image_type'])

        # resize image to image_size
        image_width, image_height = data_source['image_size'].split(' x ')
        im = image_ps.zoom(im, int(image_width), int(image_height))

        # todo maohaijun, do pretreat
        pass

        image_ps.save_image(image_path, sub_save_dir, im)
        return

    def __classify_data(self, model_dir, train_rate, valid_rate, test_rate):
        train_dir = path_join(model_dir, 'train')
        valid_dir = path_join(model_dir, 'valid')
        test_dir = path_join(model_dir, 'test')
        os.mkdir(train_dir)
        os.mkdir(valid_dir)
        os.mkdir(test_dir)
        train_file = open(path_join(train_dir, 'train.txt'), 'w')
        valid_file = open(path_join(valid_dir, 'valid.txt'), 'w')
        test_file = open(path_join(test_dir, 'test.txt'), 'w')
        verify_file = open(path_join(model_dir, 'verify.txt'), 'w')
        family = 0

        for dirname in os.listdir(model_dir):
            dir_path = path_join(model_dir, dirname)
            if os.path.isdir(dir_path) and dirname not in ['train', 'valid', 'test']:
                temp_dir = path_join(model_dir, dirname)
                sub_train_dir = path_join(train_dir, dirname)
                sub_valid_dir = path_join(valid_dir, dirname)
                sub_test_dir = path_join(test_dir, dirname)
                os.mkdir(sub_train_dir)
                os.mkdir(sub_valid_dir)
                os.mkdir(sub_test_dir)

                verify_file.write(dirname + '\n')
                for parent, dirnames, filenames in os.walk(temp_dir):
                    count = 0
                    num = len(filenames)
                    train_num = int(num * train_rate / 100)
                    train_valid_num = train_num + int(num * valid_rate / 100)
                    for filename in filenames:
                        count += 1
                        label = '/%s/%s %s\n' % (dirname, filename, family)
                        if count <= train_num:
                            shutil.copy(path_join(temp_dir, filename), sub_train_dir)
                            train_file.write('/train' + label)
                        elif train_num < count <= train_valid_num:
                            shutil.copy(path_join(temp_dir, filename), sub_valid_dir)
                            valid_file.write('/valid' + label)
                        else:
                            shutil.copy(path_join(temp_dir, filename), sub_test_dir)
                            test_file.write('/test' + label)

                    family += 1
                shutil.rmtree(dir_path)

        verify_file.close()
        train_file.close()
        valid_file.close()
        test_file.close()
        return


if __name__ == '__main__':
    print_msg("datasource name: %s job running !!!" % data_source['name'])
    RunImageProcessJob(**data_source)
