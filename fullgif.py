
import sys
import time
import pygame
pygame.display.init()
from gen import stritem_replace

VERBOSE = True


# If the gif provides 0 as a frame delay, use this instead
DEFAULT_FRAME_DELAY = 10


# between 0 and 65535
def chr16(num):
    return chr(num % 256) + chr(num / 256)


def ord16(letters):
    return ord(letters[0]) + ord(letters[1]) * 256


def check_fpsval(value):
    return value


##    print value
##    if value < 1 or value > 65535 or not isinstance(value, int):
##        raise ValueError('value must be int from 1 to 100')
def check_delayval(value):
    if value > 65535:
        return 65535
    return value


##    if value < 1 or value > 65535 or not isinstance(value, int):
##        raise ValueError('value must be int from 1 to 65535')


class Gif_Image(object):
    def __init__(self):
        self.comments = []
        self.graphics_extension_block = False
        self.image_block = None

    def clear_graphics_extension_block(self):
        self.graphics_extension_block = False
        self.user_input_required = None
        self.disposal_method = None
        self.frame_delay = None
        self.transparent_color_index = None

    def make_pygame_surface(self):
        # RGB surface
        self.surface = pygame.image.frombuffer(
            bytearray(subpixel for i in self.data for subpixel in self.color_table[i]),
            (self.width, self.height),
            'RGB'
        )
        # Palette surface
        self.surface2 = pygame.image.frombuffer(bytearray(self.data), (self.width, self.height), 'P')
        self.surface2.set_palette(self.color_table)

# Counting bits starting at 0
class Gif(object):
    """class to manipulate gifs"""

    GIF87a = 'GIF87a'
    GIF89a = 'GIF89a'

    def __init__(self, filename):
        if VERBOSE:
            print 'loading', (filename)
            start_time = time.time()
        self.images = []
        self.filename = filename
        self.data = data = open(filename, 'rb').read()
        self.version = data[:6]
        assert self.version in (self.GIF87a, self.GIF89a)
        # Where in self.data is the next piece of data, if variability is needed.
        self.tell = 6
        self.parse_headers()
        self.current_image = None
        self.parse_blocks()
        if VERBOSE:
            print 'took %.1f seconds' % (time.time() - start_time)

    def __repr__(self):
        return '<Gif: "%s" %s>' % (self.filename, self.dims)

    def parse_headers(self):
        self.dims = (ord16(self.data[6:8]), ord16(self.data[8:10]))

        screen_descriptor = ord(self.data[10])
        # bits 0-2 are the bits pixel in the image minus 1 (0-7 => 1-8)
        global_color_table_entry_size = 1 + (screen_descriptor & 7)
        # the number of entries in the global color table can be calculated as such
        self.global_color_table_entries = 1 << global_color_table_entry_size  # = 2 ** global_color_table_entry_size
        # bit 3 is whether the global color table is sorted by most used colors
        self.global_color_table_sorted = screen_descriptor & 8
        if self.global_color_table_sorted and self.version == self.GIF87a:
            self.global_color_table_sorted = 0
            # raise GIFError('global color table sorted cannot be set for %s'%self.GIF87a)
        # bits 4-6 are the bits in an entry of the original color palette minus 1 (0-7 => 1-8)
        self.color_resolution = 1 + (screen_descriptor & 112)  # & (16 + 32 + 64)
        # bit 7 is whether the global color table exists
        self.global_color_table_exists = screen_descriptor & 128

        # The index in the global color table (if it exists) for the background of the screen
        self.background_color_index = ord(self.data[11])

        # The aspect ratio of a pixel. I'm going to ignore it.
        # the ratio defines width:height
        aspect_ratio_byte = ord(self.data[12])
        if (aspect_ratio_byte):
            # This is the specific math it uses to define the ratio
            self.pixel_aspect_ratio = (aspect_ratio_byte + 15) / 64.
        else:
            # If not set then it's disabled
            self.pixel_aspect_ratio = 1
        self.tell = 13
        if self.global_color_table_exists:
            self.global_color_table = [None] * self.global_color_table_entries
            self.parse_color_table(self.global_color_table, self.global_color_table_entries)
        else:
            self.global_color_table = []

    def parse_color_table(self, table, entries):
        for i in xrange(entries):
            table[i] = (
                ord(self.data[self.tell]), ord(self.data[self.tell + 1]), ord(self.data[self.tell + 2])
            )
            self.tell += 3

    def parse_blocks(self):
        while 1:
            separator = self.data[self.tell]
            self.tell += 1
            # '\x3b' is the end of file char
            if separator == '\x3b':
                break
            if self.current_image is None:
                # graphics extension block, image block
                self.current_image = Gif_Image()
            # '\x2c' is the local image block
            if separator == '\x2c':
                self.parse_image_block()
                self.images.append(self.current_image)
                self.current_image = None
            elif separator == '\x21': #89a
                if self.version == self.GIF87a:
                    raise GIFError('87a gif has 89a block')
                label = self.data[self.tell]
                self.tell += 1
                if label == '\xf9':
                    self.parse_graphics_control_block()
                elif label == '\x01':
                    self.parse_plain_text_block()
                elif label == '\xff':
                    self.parse_application_block()
                elif label == '\xfe':
                    self.parse_comment_block()
                else:
                    raise GIFError('Unknown \\x21 block label %s|%s' % (label, ord(label)))

    def parse_image_block(self):
        self.current_image.x = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2
        self.current_image.y = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2

        self.current_image.width = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2
        self.current_image.height = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2

        packed = ord(self.data[self.tell])
        self.tell += 1
        # Bit 0 Is there a local color table
        local_color_table = packed & 1
        # Bit 1 Is the image interlaced
        self.current_image.interlaced = packed & 2
        # Bit 2 Is the local color table sorted
        self.current_image.color_table_sorted = packed & 4
        if self.current_image.color_table_sorted and self.version == self.GIF87a:
            self.current_image.color_table_sorted = 0
            # raise GIFError('color table sorted cannot be set for %s'%self.GIF87a)
        # Bits 3-4 Reserved
        # Bits 5-7 The size of the local color table (see same bits for global color table)
        color_table_entry_size = 1 + (packed & 7)
        self.current_image.color_table_entries = 1 << color_table_entry_size  # = 2 ** color_table_entry_size
        if local_color_table:
            self.current_image.color_table = [None] * self.current_image.color_table_entries
            self.parse_color_table(self.current_image.color_table, self.current_image.color_table_entries)
        else:
            # Use the global color table if there's no local one
            self.current_image.color_table = self.global_color_table
        self.current_image.data = self.parse_image_data()
        # TODO parse the data, convert to x/y lines, handle image dims / xy position,
        #  animation and overwriting/transparency
        self.current_image.make_pygame_surface()

    def parse_image_data(self):
        total = 0
        data = []
        minimum_lzw_code_size = ord(self.data[self.tell])
        self.tell += 1
        while 1:
            length = ord(self.data[self.tell])
            self.tell += 1
            if length == 0:
                parsed_data = self.parse_stream_data(minimum_lzw_code_size, data)
                return parsed_data
            total += length
            # data.extend(self.parse_stream_data(ord(i) for i in self.data[self.tell:self.tell + length]))
            data.extend([ord(i) for i in self.data[self.tell:self.tell + length]])
            self.tell += length

    def parse_stream_data(self, minimum_lzw_code_size, data):
        g = Gif_LZW(minimum_lzw_code_size, data)
        g.parse_stream_data()
        return g.stream

    def parse_graphics_control_block(self):
        block_size = ord(self.data[self.tell])
        if block_size != 4:
            raise GIFError(
                'Unexpected block size in graphics control extension block (expected 4, got %d)' % block_size
            )
        self.tell += 1
        self.current_image.graphics_extension_block = True
        packed_bit = ord(self.data[self.tell])
        self.tell += 1
        # Bit 0 Is the later color index byte has data
        has_transparent_color_index = packed_bit & 1
        # Bit 1 Is user input is required to move to the next image (ignored)
        self.current_image.user_input_required = packed_bit & 2
        # Bits 2-4 Gives one of 4 methods to dispose of the previous image
        disposal_method = packed_bit & 28  # (4 + 8 + 16)
        if disposal_method != 0 and disposal_method != 4 and disposal_method != 8 and disposal_method != 16:
            raise GIFError(
                'Previous image disposal method is invalid (expected 0, 4, 8, or 16, got %d' % disposal_method
            )
        self.current_image.disposal_method = disposal_method
        # Bits 5-7 last 3 bits are reserved
        self.current_image.frame_delay = ord16(self.data[self.tell:self.tell + 2]) or DEFAULT_FRAME_DELAY
        self.tell += 2
        if has_transparent_color_index:
            self.current_image.transparent_color_index = ord(self.data[self.tell])
        else:
            self.current_image.transparent_color_index = None
        self.tell += 1
        if self.data[self.tell] != '\x00':
            raise GIFError('Graphics control block terminator not found')

    # Ignored
    def parse_plain_text_block(self):
        # block is 15 bytes long, the first 2 are read in parse_blocks(), and last 1 is terminator
        self.tell += 12
        if self.data[self.tell] != '\x00':
            raise GIFError('Plain text block terminator not found')
        self.tell += 1
        # graphics extension block affects the next block of plain text or image type. Since we're ignoring plain text,
        #  drop the related graphics extension info
        if self.current_image.graphics_extension_block:
            self.current_image.clear_graphics_extension_block()

    # Ignored
    def parse_application_block(self):
        self.tell += 12
        data_length = self.data[self.tell]
        while 1:
            self.tell += ord(data_length) + 1
            data_length = self.data[self.tell]
            if data_length == '\x00':
                self.tell += 1
                return

    def parse_comment_block(self):
        comment_length = ord(self.data[self.tell])
        self.tell += 1
        comment = self.data[self.tell:self.tell + comment_length]
        self.current_image.comments.append(comment)
        self.tell += comment_length
        if self.data[self.tell] != '\x00':
            raise GIFError('Comment block terminator not found')
        self.tell += 1

    def get_delays(self):
        for i in self.frames:
            print ord16(self.data[i:i + 2]),

    def frame_delays(self):
        data = self.data
        self.frames = []
        self.framevals = []
        for i in xrange(len(data)):
            # 0, \x00: end previous block
            # 1-3, \x21\xf9\x04: Graphic Control Extension
            # 4, next is transparency data
            # 5-6, next 2 are frame timing data
            # 7, next is something
            # 8, next is \x00: end block
            # 9, start of next block
            # now once every 2**47 bits this will be a false positive.
            # what to do about it
            if (data[i:i + 4] == '\x00\x21\xf9\x04' and data[i + 8] == '\x00'
                    and data[i + 9] in ('\x21', '\x2c')):
                # print i+5,
                # assert data[i+9] == '\x2c' #new image block after descriptor!
                self.frames.append(i + 5)
                self.framevals.append(ord16(data[i + 5:i + 7]))
        print

    def set_fps(self, value):
        if VERBOSE:
            print 'setting FPS to', value
        value = check_fpsval(value)
        values = dict()
        value = float(value)
        last = 0
        for ind, i in enumerate(self.frames):
            frame = int(round((ind + 1) / value * 100)) - last
            last += frame
            values.setdefault(frame, [])
            values[frame].append(i)
        for i in values:
            self.set_delays(i, values[i], False)
        print 'fps set to %d' % value
        self.save()

    def set_delays(self, value, indexs=None, save=True):
        if VERBOSE:
            print 'setting delay to', value
        value = check_delayval(value)
        if indexs is None:
            if VERBOSE:
                print 'for all indexs'
            indexs = self.frames
        elif not hasattr(indexs, '__iter__'):
            if VERBOSE:
                print 'for index(s)', indexs
            indexs = [indexs]
        else:
            if VERBOSE:
                print 'for indexs', indexs
        for i in indexs:
            self.data = stritem_replace(self.data, i, chr16(value), 2)
        if save:
            self.save()

    def save(self):
        if VERBOSE:
            print 'saving', self
        open(self.filename, 'wb').write(self.data)


def main(f=None):
    if f or len(sys.argv) > 1:
        g = Gif(f or sys.argv[1])
    try:
        ind = sys.argv.index('-f')
    except ValueError:
        try:
            ind = sys.argv.index('-d')
        except ValueError:
            pass
        else:
            if sys.argv[ind + 1] == '?':
                delay = raw_input('delay (3): ') or 3
                delay = int(delay)
            else:
                delay = int(sys.argv[ind + 1])
            g.set_delays(delay)
    else:
        doFPS = True
        if sys.argv[ind + 1] == '?':
            print 'current fps: ', (100. * len(g.framevals) / sum(g.framevals))
            fps = float(raw_input('fps (30): ') or 30)
            if fps < 0:
                doFPS = False
                fps = -fps
        else:
            fps = float(sys.argv[ind + 1])
        if doFPS:
            g.set_fps(fps)
        else:
            g.set_delays(fps)
    for i in sys.argv[2:]:
        if '-O=' in i:  # force best, it's 2:25am fu
            import opt_gif
            if VERBOSE:
                print 'optimizing . . .'
            opt_gif.main(sys.argv[1])
            break
    return g


class Gif_LZW(object):
    # If the code table has reached the 2**12 limit, the code table may not be added to
    maximum_bit_size = 12

    bit_ands = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]

    def __init__(self, minimum_size, data):
        self.minimum_size = minimum_size
        self.clear_code = 1 << self.minimum_size
        self.end_of_information_code = self.clear_code + 1
        self.code_table = []

        self.reset_code_table()
        self.value_buffer = 0
        self.value_buffer_bits = 0
        self.tell = 0
        self.stream = []

        self.data = data

    def get_next_code(self):
        while self.tell < len(self.data):
            while self.value_buffer_bits < self.code_size:
                self.value_buffer += (self.data[self.tell] << self.value_buffer_bits)
                self.tell += 1
                self.value_buffer_bits += 8
            value = self.value_buffer & self.bit_ands[self.code_size]
            self.value_buffer >>= self.code_size
            self.value_buffer_bits -= self.code_size
            return value

    def parse_stream_data(self):
        self.assure_clear_code(self.get_next_code())
        prev_code = self.get_next_code()
        self.add_to_stream(prev_code)
        while self.tell < len(self.data):
            code = self.get_next_code()
            if code < self.next_code_index:
                if code == self.clear_code:
                    self.reset_code_table()
                    prev_code = None
                    continue
                if code == self.end_of_information_code:
                    return
                self.add_to_stream(code)
                if prev_code is not None:
                    self.add_to_table(prev_code, code)
            else:
                self.add_to_table(prev_code, prev_code)
                self.add_to_stream(code)
            prev_code = code

    def add_to_stream(self, code):
        self.stream.extend(self.code_table[code])

    def add_to_table(self, code, K_code):
        if self.table_immutable:
            return
        self.code_table[self.next_code_index] = self.code_table[code] + [self.code_table[K_code][0]]
        self.next_code_index += 1
        if self.next_code_index == self.next_code_table_grow:
            if self.code_size == self.maximum_bit_size:
                # Gifs aren't allowed to grow beyond this hard limit per code
                self.table_immutable = True
                return
            self.set_code_size(self.code_size + 1)

    def reset_code_table(self):
        self.code_table[:] = [None for i in xrange((1 << self.maximum_bit_size))]
        for i in xrange(1 << self.minimum_size):
            self.code_table[i] = [i]
        # self.code_table[self.clear_code] = self.clear_code
        # self.code_table[self.end_of_information_code] = self.end_of_information_code
        # Track what the next index for a code in self.code_table will be.
        self.next_code_index = self.end_of_information_code + 1
        self.set_code_size(self.minimum_size + 1)
        self.table_immutable = False

    def set_code_size(self, size):
        self.code_size = size
        self.next_code_table_grow = (1 << self.code_size)

    def assure_clear_code(self, code):
        if code != self.clear_code:
            raise GIFError('Expected clear code, got something else (%d != %d)' %(self.clear_code, code))


class GIFError(Exception):
    pass


def fit_to(start_dims, dims=(1920, 1080)):
    width, height = start_dims
    w = float(dims[0]) / width
    h = float(dims[1]) / height
    if w < h:
        w2 = dims[0]
        h2 = height * w
    else:
        w2 = width * h
        h2 = dims[1]
    return (int(float(w2)), int(float(h2)))

def display_gif(gif, fitto=(1000, 1000), loop=True):
    pygame.display.init()
    s = pygame.display.set_mode(fit_to(gif.dims, fitto))
    c = pygame.time.Clock()
    while 1:
        for i in gif.images:
            if pygame.event.get(pygame.QUIT):
                break
            s.blit(pygame.transform.smoothscale(i.surface, fit_to(i.surface.get_size(), fitto)), i.surface.get_rect())
            c.tick(100. / i.frame_delay)
            pygame.display.flip()
        else:
            if loop:
                continue
        break
    pygame.display.quit()