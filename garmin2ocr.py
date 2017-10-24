from PIL import Image
import pyocr
import cv2

import argparse
import logging
import string
import sys
import re
import os


#THRESH_ADJ = [0, -24, 24, -48, 48]
THRESH_ADJ = [0, -24, 24, -48]
LOG_FILE = 'garmin2ocr.log'


# Configure logging
# using __name__ prevents logs in other files from displaying
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

log_file = os.path.join(os.getcwd(), LOG_FILE)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
#f = ContextFilter()
#fh.addFilter(f)
logger.addHandler(fh)

logger.info('logger initialized')



# Open an image, massage, store in temp file and run OCR #
def process_image(filename, thresh):
    global cropped_image, mono_image

    logger.info('reading image \'%s\' (threshold %d)', filename, thresh)
    image = cv2.imread(filename)

    logger.debug('cropping image')
    cropped_image = image[1000:1080, 0:1300]

    logger.debug('performing image manipulation')
    mono_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    mono_image = cv2.threshold(mono_image, thresh, 255, cv2.THRESH_BINARY)[1]
    mono_image = cv2.medianBlur(mono_image, 3)
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

    # We're lazy and don't bother handling failures
    # with clean-up of these for now
    os.remove(temp_file)
    return text


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
    # The tools are returned in the recommended order of usage
    #tools = pyocr.get_available_tools()
    #tool = tools[0]
    tool = pyocr.tesseract
    logger.info('using tool \'%s\'' % tool.get_name())
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    logger.info('available languages: %s' % ', '.join(langs))
    if not 'eng' in langs:
        logger.fatal('\'eng\' not in available languages, aborting')
        sys.exit(0)

    gps_re = r'GARMIN.*\s(?P<lat>[0-9\-]{1,4}\.[0-9]{4,5})\s?(?P<long>[0-9-]{1,4}\.[0-9]{4,5}).*'
    gps_rep = re.compile(gps_re)

    # Build queue of files to process
    if args['image']:
        image_files = [args['image']]
    elif args['directory']:
        image_files = map(lambda x: os.path.join(args['directory'], x), os.listdir(args['directory']))

    # Iterate sequentially over files
    for idx, fn in enumerate(image_files):

        # Filter on filename
        if not filter_file(fn, args['prefix']):
            continue

        # Upon regex match failure, retry
        # with varying threshold inputs
        retry = 1
        while retry <= len(THRESH_ADJ):
            # Calculate current threshold for cv2
            thresh = args['threshold'] + THRESH_ADJ[retry-1]
            if thresh > 250: thresh = 250
            logger.info('processing \'%s\' [%d/%d]', fn, idx, len(image_files))
            logger.debug('threshold %d', thresh)
            logger.debug('attempt %d', retry)
            text = process_image(fn, thresh)

            fix_text = translate(text, 'SlO', '510')
            logger.debug('corrected text: \'%s\'', fix_text)
            gps_text = gps_rep.match(fix_text)

            if not gps_text:
                retry += 1
                logger.error('text not detected correctly')
                logger.warning('retrying with new threshold')
            else:
                break

        # Abort on failure after retries
        if retry > len(THRESH_ADJ):
            logger.fatal('unable to detect with higher threshold')
            import pdb
            pdb.set_trace()

            cv2.imshow('Image', cropped_image)
            cv2.imshow('Output', mono_image)
            cv2.waitKey(0)
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
    ap.add_argument('-i', '--image', type=str,
                    help='path to image')
    ap.add_argument('-d', '--directory', type=str,
                    help='path to directory containing images')
    ap.add_argument('-p', '--prefix', type=str,
                    help='prefix for images to be selected/filtered for processing')
    ap.add_argument('-t', '--threshold', type=int, default=224,
                    help='threshold value for image filtering')
    args = vars(ap.parse_args())

    # Verify input
    if not args['image'] and not args['directory']:
        logger.fatal('single image or directory containing images must be provided')
        sys.exit(1)

    main(args)

