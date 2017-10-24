from PIL import Image
import pyocr
import cv2

import colorlog
import logging

import argparse
import shutil
import sys
import re
import os


# Globals #
THRESH_ADJ = [0, -24, 24, -48]
LOG_FILE = 'garmin2ocr.log'
PROCESS_FAIL_PATH = 'failed_images'


### Configure logging ###
# Filter for our logger
class ContextFilter(logging.Filter):
    def filter(self, record):
        if 'metasync' in record.name:
            return True
        return False


CUSTOM_LOG_LEVEL = [('PIL.PngImagePlugin', logging.INFO)]

# Use colored logger for console
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = colorlog.ColoredFormatter('%(asctime)s - %(name)s:%(lineno)d - %(log_color)s%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Configure file logging
log_file = os.path.join(os.getcwd(), LOG_FILE)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
#f = ContextFilter()
#fh.addFilter(f)
logger.addHandler(fh)

# Set specific log levels for certain loggers
for customlog in CUSTOM_LOG_LEVEL:
    logging.getLogger(customlog[0]).setLevel(customlog[1])

logger.info('logger initialized')



# Open an image, massage, store in temp file and run OCR #
def process_image(filename, thresh):

    logger.info('reading image \'%s\' (threshold %d)', filename, thresh)
    image = cv2.imread(filename)

    logger.debug('cropping image')
    cropped_image = image[1000:1080, 0:1300]

    #
    # TODO: We can be more inteliigent here and take advantage of an
    # opportunity: the text is rendered in white with black outline, even
    # counting pre-processing it will be entirely monochromatic while the
    # actual view of the image will likely have very few low/zero saturation
    # pixels.
    #
    # In theory if we can seperate the two, it should isolate the text
    # very cleanly.  However, the method to do this is TBD, it's not
    # entirely straightforward.
    #
    logger.debug('performing image manipulation')
    mono_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    mono_image = cv2.threshold(mono_image, thresh, 255, cv2.THRESH_BINARY)[1]
    mono_image = cv2.medianBlur(mono_image, 3)

    # TODO: Do this better, at the very minimum use tmpfs/shared memory
    temp_file = '{}.png'.format(os.getpid())
    logger.debug('writing temporary file %s', temp_file)
    cv2.imwrite(temp_file, mono_image)

    logger.debug('opening temporary file')
    img = Image.open(temp_file)

    #text = tool.image_to_string(img, lang=lang, builder=pyocr.builders.TextBuilder())
    text = pyocr.tesseract.image_to_string(img, lang='eng', builder=pyocr.builders.TextBuilder())
    logger.info('TextBuilder output: %s', text)

    #digits = tool.image_to_string(img, lang=lang, builder=pyocr.builders.DigitBuilder())
    #logger.info('DigitBuilder: %s' % digits)

    # We're lazy and don't bother handling clean-up failures of these for now
    os.remove(temp_file)
    return (text, mono_image)


# Filter input files #
def filter_file(filename, prefix):
    if prefix and prefix not in filename:
        return False
    if '.JPG' not in filename.upper():
        return False

    return True


def translate(text, src, replace):
    tbl = dict(zip(src, replace))
    for i, c in enumerate(text):
        if c in src:
            text = text.replace(c, tbl[c])
    return text


def main(args):
    # On error, copy images to separate directory for later analysis
    failed_path = os.path.join(os.getcwd(), PROCESS_FAIL_PATH)

    # The tools are returned in the recommended order of usage
    #tools = pyocr.get_available_tools()
    #tool = tools[0]
    tool = pyocr.tesseract
    logger.info('using tool \'%s\'' % tool.get_name())

    langs = tool.get_available_languages()
    logger.info('available languages: %s' % ', '.join(langs))
    if not 'eng' in langs:
        logger.fatal('\'eng\' not in available languages, aborting')
        sys.exit(0)

    gps_re = r'GARMIN.*\s(?P<lat>[0-9\-]{1,4}\.[0-9]{4,5})\s?(?P<long>[0-9-]{1,4}\.[0-9]{4,5}).*'
    gps_rep = re.compile(gps_re)

    # Build queue of files to process
    if os.path.isfile(args['path']):
        image_files = [args['path']]
    elif os.path.isdir(args['path']):
        image_files = map(lambda x: os.path.join(args['path'], x), os.listdir(args['path']))
    else:
        logger.fatal('unable to determine if path \'%s\' is file or directory, aborting')
        sys.exit(0)

    # Iterate sequentially over files
    for idx, fn in enumerate(sorted(image_files)):

        # Filter on filename
        if not filter_file(fn, args['filter']):
            logger.debug('skipping %s [%d/%d]', fn, idx, len(image_files))
            continue

        # Upon regex match failure, retry
        # with varying threshold inputs
        retry = 1
        mono_img = []
        while retry <= len(THRESH_ADJ):

            # Calculate current threshold for cv2
            thresh = args['threshold'] + THRESH_ADJ[retry-1]
            if thresh > 250: thresh = 250

            # Do it
            logger.info('processing \'%s\' [%d/%d]', fn, idx, len(image_files))
            logger.debug('threshold %d', thresh)
            logger.debug('attempt %d', retry)
            (text, m_img) = process_image(fn, thresh)
            mono_img.append(m_img)

            # Handle common translation errors
            # '_' => ' ' still being evaluated
            fix_text = translate(text, 'SlO_', '510 ')
            logger.debug('corrected text: \'%s\'', fix_text)

            # Test against regex
            gps_text = gps_rep.match(fix_text)

            if not gps_text:
                retry += 1
                logger.error('text not detected correctly')
                if retry < len(THRESH_ADJ):
                    logger.warning('retrying with new threshold')
            else:
                break

        # Abort on failure after retries
        if retry > len(THRESH_ADJ):
            logger.fatal('unable to detect with higher threshold')

            # Copy failed source file and processed versions
            # to separate path for later error analysis
            #
            # eg: fn = '/foo/bar/image001.jpg'
            # cur_fn_base = 'image001.jpg'
            # cur_fn_base_noext = 'image001'
            # copy_failed_fn = 'failed_image/image001.jpg'
            cur_fn_base = os.path.basename(fn)
            cur_fn_base_noext = cur_fn_base.split(os.path.extsep)[0]
            copy_failed_fn = os.path.join(failed_path, cur_fn_base)

            if not os.path.isdir(failed_path):
                logger.debug('creating path \'%s\'', failed_path)
                os.mkdir(failed_path)

            logger.debug('copying failed file to \'%s\'', copy_failed_fn)
            shutil.copyfile(fn, copy_failed_fn)

            # Saved processed images
            # TODO: Record OCR'd text with each failed attempt, maybe render on image itself?
            for i in range(len(THRESH_ADJ)):
                mono_img_fn = os.path.join(failed_path, '%s_mono-%d.png' % (cur_fn_base_noext, i+1))
                logger.debug('writing failed image %s', mono_img_fn)
                cv2.imwrite(mono_img_fn, mono_img[i])

            #cv2.imshow('Image', cropped_image)
            #cv2.imshow('Output', mono_image)

            #import pdb
            #pdb.set_trace()

            #cv2.waitKey(0)
            sys.exit(0)

        logger.info('detected coordinates:')
        logger.info('(%s, %s)', gps_text.group('lat'), gps_text.group('long'))

        # TODO: log/store to disk
        # prompt for user validation
        # write gps data to exif


# void main(args) #
if __name__ == '__main__':

    # Handle args
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', type=str, required=True,
                    help='path to image or directory containing images')
    ap.add_argument('-f', '--filter', type=str,
                    help='prefix for images to be selected/filtered for processing')
    ap.add_argument('-t', '--threshold', type=int, default=224,
                    help='threshold value for image filtering')
    args = vars(ap.parse_args())

    # main #
    main(args)

