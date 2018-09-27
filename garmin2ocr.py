# pylint: disable=no-member
# pylint is not detecting cv2 opencv library members #
"""
This tool performs OCR on images gathered from a dashcam with
GPS data overlaid, determines the coordinates and embed with
the exif metadata

It will analyze additional photo collections taken during an overlapping
time period, perform grouping where applicable and edit exif capture
time with timezone adjustment

Further ideas may include interpolation of GPS coordinates for non-tagged
photos if local grouping is sufficiently effective
"""

from datetime import datetime

import argparse
import shutil
import sys
import re
import os
import io

import logging
import colorlog

from PIL import Image
import pyocr
import cv2

import piexif

import numpy as np

import gpxpy
import gpxpy.gpx

import tre


### Globals ###
# Threshold adjustment values to try (first is default)
DEFAULT_THRESH = 224
THRESH_ADJ = [0, -24, 24, -48]

LOG_FILE = 'garmin2ocr.log'
GPX_OUT = 'dashcam.gpx'

EXIF_DATETIME_FMT = '%Y:%m:%d %H:%M:%S'

# Location to save failed processing attempts
IMG_FAILED_PATH = 'images_failed'
IMG_SUCCEED_PATH = 'images_processed'

# RegEx to match our dashcam GPS coordinates
GPS_RE = r'GARMIN.*\s(?P<lat>[0-9\-]{1,4}\.[0-9]{4,5})\s?(?P<long>[0-9-]{1,4}\.[0-9]{4,5}).*'
#GPS_TRE = r'GARMIN\s([0-9]{2}\/[0-9]{2}\/[0-9]{4})\s?([0-9]{2}:[0-9]{2}:[0-9]{2}) ([AP]M).* ([0-9\-]{1,4}\.[0-9]{4,5})\s?([0-9-]{1,4}\.[0-9]{4,5}).*'
GPS_TRE = r'GARMIN\s([0-9]{2}\/[0-9]{2}\/[0-9]{4})\s?([0-9]{2}:[0-9]{2}:[0-9]{2}).* ([0-9\-]{1,4}\.[0-9]{4,5})\s?([0-9-]{1,4}\.[0-9]{4,5}).*'


### Configure logging ###
# Filter for our logger
# pylint: disable=too-few-public-methods
#class ContextFilter(logging.Filter):
#    """ Custom filter based on record.name """
#    def filter(self, record):
#        if 'metasync' in record.name:
#            return True
#        return False


CUSTOM_LOG_LEVEL = [('PIL.PngImagePlugin', logging.INFO)]

# Use colored logger for console
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
stream_h = logging.StreamHandler()
stream_h.setLevel(logging.DEBUG)
formatter = colorlog.ColoredFormatter(
    '%(asctime)s - %(name)s:%(lineno)d - %(log_color)s%(levelname)s - %(message)s'
)
stream_h.setFormatter(formatter)
logger.addHandler(stream_h)

# Configure file logging
# TODO: Add single log per invocation
log_file = os.path.join(os.getcwd(), LOG_FILE)
file_h = logging.FileHandler(log_file)
file_h.setLevel(logging.DEBUG)
file_h.setFormatter(formatter)
# f = ContextFilter()
# file_h.addFilter(f)
logger.addHandler(file_h)

# Set specific log levels for certain loggers
for customlog in CUSTOM_LOG_LEVEL:
    logging.getLogger(customlog[0]).setLevel(customlog[1])

logger.info('logger initialized')


def process_image(filename, thresh, enhanced=False):
    """ Open an image, massage, store in temp file and run OCR """

    logger.info('reading image \'%s\' (threshold %d)', filename, thresh)
    image = cv2.imread(filename)

    logger.debug('cropping image')
    cropped_image = image[1030:1070, 0:1300]

    logger.debug('performing image manipulation')
    mono_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    mono_image = cv2.threshold(mono_image, thresh, 255, cv2.THRESH_BINARY)[1]
    mono_image = cv2.medianBlur(mono_image, 3)

    #
    # If first pass of OCR fails, re-process image
    #
    # Our goal is isolate out the monochromatic data
    # with a mask: most of image has various levels of
    # saturation, the GPS data text is rendered in
    # monochrome.  We will also combine this masked
    # image with the first processing attempt to
    # remove noise.
    #
    if enhanced:
        blur_image = cv2.medianBlur(cropped_image, 3)
        hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
        black = np.array([0, 0, 0])
        white = np.array([179, 10, 255])
        mask = cv2.inRange(hsv_image, black, white)
        masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        mono_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        combined_image = cv2.addWeighted(mono_masked_image, 0.5, mono_image, 0.5, 0.0)

        #mask_file = '{}-masked.png'.format(os.getpid())
        #logger.debug('writing temporary file %s', mask_file)
        #cv2.imwrite(mask_file, combined_image)

        mono_image = combined_image

    # TODO: Do this better, at the very minimum use tmpfs/shared memory
    temp_file = '{}.png'.format(os.getpid())
    logger.debug('writing temporary file %s', temp_file)
    cv2.imwrite(temp_file, mono_image)

    logger.debug('opening temporary file')
    img = Image.open(temp_file)

    # text = tool.image_to_string(img, lang=lang, builder=pyocr.builders.TextBuilder())
    logger.debug('running tesseract')
    text = pyocr.tesseract.image_to_string(img, lang='eng', builder=pyocr.builders.TextBuilder())
    logger.info('TextBuilder output: %s', text)

    # digits = tool.image_to_string(img, lang=lang, builder=pyocr.builders.DigitBuilder())
    # logger.info('DigitBuilder: %s' % digits)

    # We're lazy and don't bother handling clean-up failures of these for now
    os.remove(temp_file)
    return (text, mono_image)


def filter_file(filename, prefix):
    """ Filter input files """
    if prefix and prefix not in filename:
        return False
    if '.JPG' not in filename.upper():
        return False

    return True


def translate(text, src, replace):
    """ Quick hack to search & replace chars, very ugly """

    tbl = dict(zip(src, replace))
    for char in text:
        if char in src:
            text = text.replace(char, tbl[char])
    return text


def show_exif(fname):

    exif_dict = piexif.load(fname)
    for ifd_name in exif_dict:
        logger.debug("{0} IFD:".format(ifd_name))
        for key in exif_dict[ifd_name]:
            try:
                logger.debug('  %s: %s', piexif.TAGS[ifd_name][key]['name'], exif_dict[ifd_name][key])
            except:
                logger.error('unable to index key %s', key)
                break


# pylint: disable=too-many-locals
def process_files(args, image_files):
    """ Iterate through list of files and process """

    # Init #
    gps_rep = re.compile(GPS_RE)
    gps_trep = tre.compile(GPS_TRE, tre.EXTENDED)
    fz = tre.Fuzzyness(maxerr = 3)
    output_data = open('garmin2ocr.out', 'w')

    # Create first track in our GPX:
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    succeed_path = os.path.join(os.getcwd(), IMG_SUCCEED_PATH)
    if not os.path.isdir(succeed_path):
        logger.debug('creating path \'%s\'', succeed_path)
        os.mkdir(succeed_path)

    failed_path = os.path.join(os.getcwd(), IMG_FAILED_PATH)
    if not os.path.isdir(failed_path):
        logger.debug('creating path \'%s\'', failed_path)
        os.mkdir(failed_path)

    # Iterate sequentially over files
    for idx, fname in enumerate(sorted(image_files)):

        cur_fn_base = os.path.basename(fname)
        succeed_fn = os.path.join(succeed_path, cur_fn_base)

        # Filter on filename
        if not filter_file(fname, args['filter']):
            logger.debug('skipping %s [%d/%d]', fname, idx, len(image_files))
            continue

        if os.path.exists(succeed_fn):
            logger.debug('skipping existing processed file found at %s', succeed_fn)
            continue

        # Upon regex match failure, retry
        # with varying threshold inputs
        retry = 1

        # Keep copies of image processing,
        # attempts reset after successful
        mono_img = []
        while retry <= len(THRESH_ADJ):

            # Calculate current threshold for cv2
            thresh = args['threshold'] + THRESH_ADJ[retry - 1]
            if thresh > 250:
                thresh = 250

            # Do it
            logger.info('processing \'%s\' [%d/%d]', fname, idx, len(image_files))
            logger.debug('threshold %d', thresh)
            logger.debug('attempt %d', retry)
            (text, m_img) = process_image(fname, thresh, enhanced=True)
            mono_img.append(m_img)

            # Handle common translation errors
            # '_' => ' ' still being evaluated
            fix_text = translate(text, 'SlO_,', '510 .')
            logger.debug('corrected text: \'%s\'', fix_text)

            # Test against regex
            gps_text = gps_trep.search(fix_text, fz)

            # This needs to be improved
            if gps_text:
                (date_str, time_str, gps_lat_str, gps_long_str) = map(lambda x: gps_text[x], range(1,5))
                if len(time_str) == 8:
                    time_str = '%s:%s:%s' % (time_str[0:2], time_str[3:5], time_str[6:8])
                else:
                    gps_text = None

                try:
                    if abs(float(gps_lat_str)) > 90:
                        gps_text = None

                    if abs(float(gps_long_str)) > 180:
                        gps_text = None
                except:
                    gps_text = None

                try:
                    image_ts = datetime.strptime('%s %s' % (date_str, time_str), '%m/%d/%Y %H:%M:%S')
                    logger.info('timestamp parsed: %s', image_ts)
                except ValueError:
                    gps_text = None

                #logger.debug(gps_text.groups())
                #for idx, grp in enumerate(gps_text.groups()):
                #    logger.debug(gps_text[idx])

            #import pdb
            #pdb.set_trace()

            if not gps_text:
                retry += 1
                logger.warning('text not detected correctly')
                if retry < len(THRESH_ADJ):
                    logger.warning('retrying with new threshold')
            else:
                break

        # Abort on failure after retries
        if retry > len(THRESH_ADJ):
            logger.error('unable to detect with higher threshold')

            # Copy failed source file and processed versions
            # to separate path for later error analysis
            #
            # eg: fname = '/foo/bar/image001.jpg'
            # cur_fn_base = 'image001.jpg'
            # cur_fn_base_noext = 'image001'
            # copy_failed_fn = 'failed_image/image001.jpg'
            #cur_fn_base = os.path.basename(fname)
            cur_fn_base_noext = cur_fn_base.split(os.path.extsep)[0]
            copy_failed_fn = os.path.join(failed_path, cur_fn_base)

            logger.debug('copying failed file to \'%s\'', copy_failed_fn)
            shutil.copyfile(fname, copy_failed_fn)

            # Saved processed images
            # TODO: Record OCR'd text with each failed attempt, maybe render on image itself?
            for i in range(len(THRESH_ADJ)):
                mono_img_fn = '%s_mono-%d.png' % (cur_fn_base_noext, i + 1)
                mono_img_path = os.path.join(failed_path, mono_img_fn)
                logger.debug('writing failed image %s', mono_img_path)
                cv2.imwrite(mono_img_path, mono_img[i])

            # cv2.imshow('Image', cropped_image)
            # cv2.imshow('Output', mono_image)

            # import pdb
            # pdb.set_trace()

            # cv2.waitKey(0)
            #sys.exit(0)
            continue

        (gps_lat, gps_long) = (float(gps_lat_str), float(gps_long_str))
        coord_text = '(%s, %s)' % (gps_lat_str, gps_long_str)
        logger.info('detected coordinates:')
        logger.info(coord_text)

        # Write to log and append point to GPX track
        output_data.write('%s : %s\n' % (fname, coord_text))
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(gps_lat_str, gps_long_str, time=image_ts))

        logger.debug('copying file to \'%s\'', succeed_fn)
        shutil.copyfile(fname, succeed_fn)
        #show_exif(succeed_fn)

        # Update the EXIF on the new copy
        current_exif = piexif.load(succeed_fn)
        zeroith_ifd = current_exif['0th']
        exif_ifd = current_exif['Exif']

        o = io.BytesIO()
        thumb_im = Image.open(fname)
        thumb_im.thumbnail((50, 50), Image.ANTIALIAS)
        thumb_im.save(o, "jpeg")
        thumbnail = o.getvalue()

        zeroith_ifd[piexif.ImageIFD.Software] = u"piexif"
        zeroith_ifd[piexif.ImageIFD.XResolution] = (72, 1)
        zeroith_ifd[piexif.ImageIFD.YResolution] = (72, 1)
        zeroith_ifd[piexif.ImageIFD.DateTime] = datetime.strftime(image_ts, EXIF_DATETIME_FMT)
        zeroith_ifd[piexif.ImageIFD.ImageDescription] = ""

        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = datetime.strftime(image_ts, EXIF_DATETIME_FMT)
        exif_ifd[piexif.ExifIFD.DateTimeDigitized] = datetime.strftime(image_ts, EXIF_DATETIME_FMT)

        gps_lat_deg = int(abs(gps_lat))
        gps_lat_min = (abs(gps_lat) - gps_lat_deg) * 60
        gps_lat_dir = 'N' if gps_lat > 0 else 'S'

        gps_long_deg = int(abs(gps_long))
        gps_long_min = (abs(gps_long) - gps_long_deg) * 60
        gps_long_dir = 'E' if gps_long > 0 else 'W'

        logger.debug('calculated GPS: (%s %s), (%s %s)', gps_lat_deg, gps_lat_min, gps_long_deg, gps_long_min)
        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 2, 0, 0),
            piexif.GPSIFD.GPSAltitudeRef: 1,
            piexif.GPSIFD.GPSLatitudeRef: gps_lat_dir,
            piexif.GPSIFD.GPSLatitude: ((gps_lat_deg, 1), (gps_lat_min*10000, 10000), (0, 1)),
            piexif.GPSIFD.GPSLongitudeRef: gps_long_dir,
            piexif.GPSIFD.GPSLongitude: ((gps_long_deg, 1), (gps_long_min*10000, 10000), (0, 1)),
            piexif.GPSIFD.GPSDateStamp: datetime.strftime(image_ts, EXIF_DATETIME_FMT)
            #piexif.GPSIFD.GPSDateStamp: timestamp
        }

        first_ifd = {
            piexif.ImageIFD.Make: u"Garmin",
            piexif.ImageIFD.XResolution: (40, 1),
            piexif.ImageIFD.YResolution: (40, 1),
            piexif.ImageIFD.Compression: 6,
            piexif.ImageIFD.Software: u"piexif",
            piexif.ImageIFD.ImageDescription: "",
            piexif.ImageIFD.UniqueCameraModel: zeroith_ifd[piexif.ImageIFD.Model],
            piexif.ImageIFD.DateTime: datetime.strftime(image_ts, '%Y-%m-%d %H:%M:%S')
        }

        # Compile EXIF and write to file
        exif_dict = {"0th": zeroith_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": first_ifd, "thumbnail": thumbnail}
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, succeed_fn)

    with open(GPX_OUT, 'w') as gpx_file:
        gpx_file.write(gpx.to_xml(version='1.1'))
        logger.info('write gpx data to %s', GPX_OUT)


def main(args):
    """ main() """

    # The tools are returned in the recommended order of usage
    # tools = pyocr.get_available_tools()
    # tool = tools[0]
    tool = pyocr.tesseract
    logger.info('using tool \'%s\'', tool.get_name())

    langs = tool.get_available_languages()
    logger.info('available languages: %s', ', '.join(langs))
    if 'eng' not in langs:
        logger.fatal('\'eng\' not in available languages, aborting')
        sys.exit(0)

    # Build queue of files to process
    if os.path.isfile(args['path']):
        image_files = [args['path']]
    elif os.path.isdir(args['path']):
        image_files = map(lambda x: os.path.join(args['path'], x), os.listdir(args['path']))
    else:
        logger.fatal('unable to determine if path \'%s\' is file or directory, aborting')
        sys.exit(0)

    process_files(args, image_files)


# void main(args) #
if __name__ == '__main__':

    # Handle args
    argp = argparse.ArgumentParser()
    argp.add_argument('-p', '--path', type=str, required=True,
                      help='path to image or directory containing images')
    argp.add_argument('-f', '--filter', type=str,
                      help='prefix for images to be selected/filtered for processing')
    argp.add_argument('-t', '--threshold', type=int, default=DEFAULT_THRESH,
                      help='threshold value for image filtering')
    arguments = vars(argp.parse_args())

    # main #
    main(arguments)
