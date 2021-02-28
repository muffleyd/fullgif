import sys
import time
import pygame
from dmgen.gen import stritem_replace

pygame.display.init()

VERBOSE = False
USABLE = True
ENFORCE_VERSION = False
COERCE_DISPOSAL_METHOD = True

# If the gif provides 0 as a frame delay, use this instead
DEFAULT_FRAME_DELAY = 10
# The minimum allowed frame delay.  Any non-zero lower frame delay will be set to this instead.
MINIMUM_FRAME_DELAY = 1


# between 0 and 65535
def chr16(num):
    return chr(num % 256) + chr(num // 256)


def ord16(letters):
    return letters[0] + letters[1] * 256


def check_fpsval(value):
    return value


def check_delayval(value):
    if value > 65535:
        return 65535
    return value


class Gif_Image(object):
    def __init__(self):
        self.comments = []
        self.graphics_extension_block = False
        self.user_input_required = None
        self.disposal_method = None
        self.frame_delay = DEFAULT_FRAME_DELAY
        self.transparent_color_index = None
        self.image_block = None
        self.image = None
        self.rect = None
        self.x = None
        self.y = None
        self.width = None
        self.height = None
        self.data = None
        self.decompressed_data = None
        self.interlaced = None
        self.color_table = None

    def set_frame_delay(self, frame_delay):
        self.frame_delay = frame_delay or DEFAULT_FRAME_DELAY
        # Enforce the minimum frame delay
        if self.frame_delay < MINIMUM_FRAME_DELAY:
            self.frame_delay = MINIMUM_FRAME_DELAY

    def clear_graphics_extension_block(self):
        self.graphics_extension_block = False
        self.user_input_required = None
        self.disposal_method = None
        self.frame_delay = None
        self.transparent_color_index = None

    def set_decompressed_data(self, data):
        self.decompressed_data = data
        self.data = None

    def decompress_data(self):
        if not self.decompressed_data and self.data:
            self.set_decompressed_data(self.data.parse_stream_data())

    def deinterlace(self):
        # The rows of an Interlaced image are arranged in the following order:
        #       Group 1 : Every 8th row, starting with row 0.
        #       Group 2 : Every 8th row, starting with row 4.
        #       Group 3 : Every 4th row, starting with row 2.
        #       Group 4 : Every 2nd row, starting with row 1.
        new_data = bytearray(len(self.decompressed_data))
        original_row = 0
        for interlace_range in (
                range(0, self.height, 8),
                range(4, self.height, 8),
                range(2, self.height, 4),
                range(1, self.height, 2)
        ):
            for new_row in interlace_range:
                self.add_deinterlaced_row(new_row, original_row, new_data)
                original_row += 1
        self.decompressed_data = new_data[:len(self.decompressed_data)]

    def add_deinterlaced_row(self, new_row, original_row, new_data):
        # Copy the data over from the original data one row at a time.
        new_data[new_row * self.width:new_row * self.width + self.width] = \
            self.decompressed_data[original_row * self.width:original_row * self.width + self.width]

    def make_pygame_surface(self):
        if self.image:
            return
        if not self.decompressed_data:
            self.decompress_data()
        # If the data is too short, append transparent color indexes, or default to 0, until it's the right length.
        if len(self.decompressed_data) < self.width * self.height:
            if self.transparent_color_index is None:
                color_index = 0
            else:
                color_index = self.transparent_color_index
            self.decompressed_data += bytes(
                [color_index] * (self.width * self.height - len(self.decompressed_data))
            )
        if self.interlaced:
            self.deinterlace()
        self.image = pygame.image.frombuffer(self.decompressed_data, (self.width, self.height), 'P')
        if self.transparent_color_index is not None:
            # We're using colorkey transparency, so if the transparent color appears elsewhere in the color table
            #  it could mean the color appears in other pixels.  Find an unused color value.
            if self.color_table.count(self.color_table[self.transparent_color_index]) > 1:
                for b in range(256):
                    transparent = bytes((0, 0, b))
                    if not self.color_table.count(transparent):
                        break
                else:
                    # Only 256 possible values in the color table, so if we're here then this one can't exist in it.
                    transparent = bytes((0, 1, 0))
                # The color table could be the same object as the global color table, so make a copy.
                self.color_table = self.color_table[:]
                self.color_table[self.transparent_color_index] = transparent
        self.image.set_palette(self.color_table)
        self.image = self.image.convert(24)
        if self.transparent_color_index is not None:
            self.image.set_colorkey(self.color_table[self.transparent_color_index])
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)


# Counting bits starting at 0
class Gif(object):
    """class to manipulate gifs"""

    GIF87a = b'GIF87a'
    GIF89a = b'GIF89a'

    def __init__(self, filename, decompress=False, make_surfaces=False):
        if VERBOSE:
            print('loading', filename)
            start_time = time.time()
        self.decompress = decompress
        self.make_surfaces = make_surfaces
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
            print('took %.2f seconds' % (time.time() - start_time))

    def __repr__(self):
        return '<Gif: "%s" %s>' % (self.filename, self.dims)

    def parse_headers(self):
        self.dims = (ord16(self.data[6:8]), ord16(self.data[8:10]))

        screen_descriptor = self.data[10]
        # bits 0-2 are the bits pixel in the image minus 1 (0-7 => 1-8)
        global_color_table_entry_size = 1 + (screen_descriptor & 7)
        # the number of entries in the global color table can be calculated as such
        global_color_table_entries = 1 << global_color_table_entry_size  # = 2 ** global_color_table_entry_size
        # bit 3 is whether the global color table is sorted by most used colors
        self.global_color_table_sorted = screen_descriptor & 8
        if self.global_color_table_sorted and self.version == self.GIF87a:
            self.global_color_table_sorted = 0
        # bits 4-6 are the bits in an entry of the original color palette minus 1 (0-7 => 1-8)
        self.color_resolution = 1 + (screen_descriptor >> 4 & 7)  # bitshift to go to bits 0-2, then (1 + 2 + 4)
        # bit 7 is whether the global color table exists
        self.global_color_table_exists = bool(screen_descriptor & 128)

        # The index in the global color table (if it exists) for the background of the screen
        self.background_color_index = self.data[11]

        # The aspect ratio of a pixel. I'm going to ignore it.
        # the ratio defines width:height
        aspect_ratio_byte = self.data[12]
        if aspect_ratio_byte:
            # This is the specific math it uses to define the ratio
            self.pixel_aspect_ratio = (aspect_ratio_byte + 15) / 64
        else:
            # If not set then it's disabled
            self.pixel_aspect_ratio = 1
        self.tell = 13
        if self.global_color_table_exists:
            self.global_color_table = [None] * global_color_table_entries
            self.parse_color_table(self.global_color_table)
        else:
            self.global_color_table = []

    def parse_color_table(self, table):
        for i in range(len(table)):
            table[i] = self.data[self.tell:self.tell + 3]
            self.tell += 3

    def parse_blocks(self):
        while 1:
            separator = self.data[self.tell]
            self.tell += 1
            # '\x3b' = 59 is the end of file char
            if separator == 59:
                break
            if self.current_image is None:
                # graphics extension block, image block
                self.current_image = Gif_Image()
            # '\x2c' = 44 is the local image block
            if separator == 44:
                self.parse_image_block()
                self.images.append(self.current_image)
                self.current_image = None
            elif separator == 33:  # 89a '\x21' = 33
                if ENFORCE_VERSION and self.version == self.GIF87a:
                    raise GIFError('87a gif has 89a block')
                label = self.data[self.tell]
                self.tell += 1
                if label == 249:  # '\xf9' = 249
                    self.parse_graphics_control_block()
                elif label == 1:  # '\x01' = 1
                    self.parse_plain_text_block()
                elif label == 255:  # '\xff' = 255
                    self.parse_application_block()
                elif label == 254:  # '\xfe' = 254
                    self.parse_comment_block()
                else:
                    raise GIFError('Unknown \\x21/33 block label %s' % label)
            else:
                raise GIFError('Unknown separator label %s' % separator)

    def parse_image_block(self):
        self.current_image.x = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2
        self.current_image.y = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2

        self.current_image.width = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2
        self.current_image.height = ord16(self.data[self.tell:self.tell + 2])
        self.tell += 2

        packed = self.data[self.tell]
        self.tell += 1
        # Bits 0-2 The size of the local color table (see same bits for global color table)
        color_table_entry_size = 1 + (packed & 7)  # (1 + 2 + 4)
        color_table_entries = 1 << color_table_entry_size  # = 2 ** color_table_entry_size
        # Bits 3-4 Reserved
        # Bit 5 Is the local color table sorted
        self.current_image.color_table_sorted = packed & 32
        # Sorted color table isn't allowed in 87a
        if self.current_image.color_table_sorted and ENFORCE_VERSION and self.version == self.GIF87a:
            self.current_image.color_table_sorted = 0
        # Bit 6 Is the image interlaced
        self.current_image.interlaced = bool(packed & 64)
        # Bit 7 Is there a local color table
        local_color_table = bool(packed & 128)

        if local_color_table:
            self.current_image.color_table = [None] * color_table_entries
            self.parse_color_table(self.current_image.color_table)
        else:
            # Use the global color table if there's no local one
            self.current_image.color_table = self.global_color_table
        self.current_image.data = self.parse_image_data()
        if self.decompress:
            self.current_image.decompress_data()
        # TODO parse the data, convert to x/y lines, handle image dims / xy position,
        #  animation and overwriting/transparency
        if self.make_surfaces:
            self.current_image.make_pygame_surface()

    def parse_image_data(self):
        total = 0
        data = bytearray()
        minimum_lzw_code_size = self.data[self.tell]
        self.tell += 1
        while 1:
            length = self.data[self.tell]
            self.tell += 1
            if length == 0:
                return Gif_LZW(minimum_lzw_code_size, data)
            total += length
            data += self.data[self.tell:self.tell + length]
            self.tell += length

    def parse_stream_data(self, minimum_lzw_code_size, data):
        g = Gif_LZW(minimum_lzw_code_size, data)
        return g.parse_stream_data()

    def parse_graphics_control_block(self):
        block_size = self.data[self.tell]
        if block_size != 4:
            raise GIFError(
                'Unexpected block size in graphics control extension block (expected 4, got %d)' % block_size
            )
        self.tell += 1
        self.current_image.graphics_extension_block = True
        packed_bit = self.data[self.tell]
        self.tell += 1
        # Bit 0 Is the later color index byte has data
        has_transparent_color_index = packed_bit & 1
        # Bit 1 Is user input is required to move to the next image (ignored)
        self.current_image.user_input_required = packed_bit & 2
        # Bits 2-4 Gives one of 4 methods to dispose of the previous image
        disposal_method = packed_bit >> 2 & 7  # bitshift to go to bits 0-2, then (1 + 2 + 4)
        if disposal_method > 3:
            if COERCE_DISPOSAL_METHOD:
                disposal_method = 0
            else:
                raise GIFError(
                    'Previous image disposal method is invalid (expected 0, 1, 2, or 3, got %d' % disposal_method
                )
        self.current_image.disposal_method = disposal_method
        # Bits 5-7 last 3 bits are reserved
        # Set frame delay, or use the default if it's zero.
        self.current_image.set_frame_delay(ord16(self.data[self.tell:self.tell + 2]))
        self.tell += 2
        if has_transparent_color_index:
            self.current_image.transparent_color_index = self.data[self.tell]
        else:
            self.current_image.transparent_color_index = None
        self.tell += 1
        if self.data[self.tell] != 0:
            raise GIFError('Graphics control block terminator not found')
        self.tell += 1

    # Ignored
    def parse_plain_text_block(self):
        # block is 15 bytes long, the first 2 are read in parse_blocks(), and last 1 is terminator
        self.tell += 12
        if self.data[self.tell] != 0:
            raise GIFError('Plain text block terminator not found')
        self.tell += 1
        # graphics extension block affects the next block of plain text or image type. Since we're ignoring plain text,
        #  drop the related graphics extension info
        if self.current_image.graphics_extension_block:
            self.current_image.clear_graphics_extension_block()

    # Ignored
    def parse_application_block(self):
        # Block header indicates this is 11 bytes long, so jump forward this 1 byte and those 11.
        self.tell += 12
        data_length = self.data[self.tell]
        # Process sub-blocks until a 0.
        while data_length:
            self.tell += data_length + 1
            data_length = self.data[self.tell]
        self.tell += 1

    def parse_comment_block(self):
        # Process sub-blocks until a 0.
        while 1:
            comment_length = self.data[self.tell]
            self.tell += 1
            if not comment_length:
                break
            self.current_image.comments.append(self.data[self.tell:self.tell + comment_length])
            self.tell += comment_length

    def get_delays(self):
        for i in self.frames:
            print(ord16(self.data[i:i + 2]), end=' ')

    def frame_delays(self):
        data = self.data
        self.frames = []
        self.framevals = []
        for i in range(len(data)):
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
        print()

    def set_fps(self, value):
        if VERBOSE:
            print('setting FPS to', value)
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
        print('fps set to %d' % value)
        self.save()

    def set_delays(self, value, indexs=None, save=True):
        if VERBOSE:
            print('setting delay to', value)
        value = check_delayval(value)
        if indexs is None:
            if VERBOSE:
                print('for all indexs')
            indexs = self.frames
        elif not hasattr(indexs, '__iter__'):
            if VERBOSE:
                print('for index(s)', indexs)
            indexs = [indexs]
        else:
            if VERBOSE:
                print('for indexs', indexs)
        for i in indexs:
            self.data = stritem_replace(self.data, i, chr16(value), 2)
        if save:
            self.save()

    def save(self):
        if VERBOSE:
            print('saving', self)
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
                delay = input('delay (3): ') or 3
                delay = int(delay)
            else:
                delay = int(sys.argv[ind + 1])
            g.set_delays(delay)
    else:
        doFPS = True
        if sys.argv[ind + 1] == '?':
            print('current fps: ', (100 * len(g.framevals) / sum(g.framevals)))
            fps = float(input('fps (30): ') or 30)
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
            from dmgen import opt_gif
            if VERBOSE:
                print('optimizing . . .')
            opt_gif.main(sys.argv[1])
            break
    return g


class Gif_LZW(object):
    # If the code table has reached the 2**12 limit, the code table may not be added to
    maximum_bit_size = 12

    bit_ands = [2 ** i - 1 for i in range(13)]

    def __init__(self, minimum_size, data):
        self.minimum_size = minimum_size
        self.clear_code = 1 << self.minimum_size
        self.end_of_information_code = self.clear_code + 1
        self.code_table = []

        # The code_table will be reset to the inside of reset_code_table.
        self.default_code_table = [None] * (1 << self.maximum_bit_size)
        self.default_code_table[:1 << minimum_size] = [bytes([i]) for i in range(1 << minimum_size)]

        self.reset_code_table()
        self.stream = bytearray()

        self.data = data

    def _get_next_code(self):
        value_buffer = 0
        value_buffer_bits = 0
        bit_ands = self.bit_ands
        code_size = self.code_size
        for byte in self.data:
            if value_buffer_bits >= code_size:
                value = value_buffer & bit_ands[code_size]
                value_buffer >>= code_size
                value_buffer_bits -= code_size
                yield value
                code_size = self.code_size
            value_buffer += (byte << value_buffer_bits)
            value_buffer_bits += 8

    def parse_stream_data(self):
        self.get_next_code = self._get_next_code()
        # Some gifs don't respect this part of the standard, oh well.
        # self.assure_clear_code(next(self.get_next_code))
        while 1:
            response = self._parse_stream_data()
            # Fake tail recursion by returning 1.
            if response == 1:
                continue
            else:
                break
        return self.stream

    def _parse_stream_data(self):
        next_code_index = self.next_code_index
        code_table = self.code_table
        clear_code = self.clear_code
        end_of_information_code = self.end_of_information_code
        table_immutable = False
        prev_code = clear_code
        # clear codes can appear AT ANY TIME
        while prev_code == clear_code:
            try:
                prev_code = next(self.get_next_code)
            except StopIteration:
                return 0
            if prev_code == end_of_information_code:
                return 0
        self.stream += code_table[prev_code]
        for code in self.get_next_code:
            if code < next_code_index:
                if code == clear_code:
                    self.reset_code_table()
                    # No tail recursion, so here we are. Wipe out prev_code like this.
                    return 1
                if code == end_of_information_code:
                    return 0
                K_code = code
            else:
                K_code = prev_code

            if not table_immutable:
                code_table[next_code_index] = code_table[prev_code] + bytes([code_table[K_code][0]])
                next_code_index += 1
                if next_code_index == self.next_code_table_grow:
                    if self.code_size == self.maximum_bit_size:
                        # Gifs aren't allowed to grow beyond this hard limit per code
                        table_immutable = True
                    else:
                        self.set_code_size(self.code_size + 1)

            self.stream += code_table[code]
            prev_code = code

    def reset_code_table(self):
        self.code_table[:] = self.default_code_table[:]
        # Track what the next index for a code in self.code_table will be.
        self.next_code_index = self.end_of_information_code + 1
        self.set_code_size(self.minimum_size + 1)

    def set_code_size(self, size):
        self.code_size = size
        self.next_code_table_grow = (1 << self.code_size)

    def assure_clear_code(self, code):
        if code != self.clear_code:
            raise GIFError('Expected clear code, got something else (%d != %d)' % (self.clear_code, code))


class GIFError(Exception):
    pass


def fit_to(start_dims, dims=(1920, 1080)):
    width, height = start_dims
    w = dims[0] / width
    h = dims[1] / height
    if w < h:
        w2 = dims[0]
        h2 = height * w
    else:
        w2 = width * h
        h2 = dims[1]
    return (int(float(w2)), int(float(h2)))


def display_gif(gif, fitto=(1000, 1000), loop=True):
    pygame.display.init()
    base_s = pygame.Surface(gif.dims)
    display_dims = fit_to(gif.dims, fitto)
    s = pygame.display.set_mode(display_dims)
    s_rect = s.get_rect()
    if gif.background_color_index:
        base_s.fill(gif.global_color_table[gif.background_color_index])
        pygame.display.flip()
    c = pygame.time.Clock()
    while 1:
        frame_delay = 0
        for i in gif.images:
            if not i.image:
                i.make_pygame_surface()
            if pygame.event.get(pygame.QUIT):
                break
            base_s.blit(i.image, i.rect)
            s.blit(pygame.transform.smoothscale(base_s, display_dims), s_rect)
            if frame_delay:
                c.tick(100. / frame_delay)
            pygame.display.flip()
            frame_delay = i.frame_delay
        else:
            if loop:
                if frame_delay:
                    c.tick(100. / frame_delay)
                continue
        break
    pygame.display.quit()


def explode_gif(gif, output_folder):
    import os
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for ind, i in enumerate(gif.images):
        pygame.image.save(i.image, os.path.join(output_folder, '%d.png' % (ind + 1)))


# Threaded decompression, doesn't speed up with the python implementation of LZW.
def decompress_gif(g):
    from dmgen import threaded_worker, gen, cores
    with gen.timer():
        with threaded_worker.threaded_worker(threads=max(1, cores.CORES - 1)) as tw:
            for i in g.images:
                tw.put(i.decompress_data)
            for i in g.images:
                tw.get()


# Multiprocessed decompression, does speed up with the python implementation of LZW.
def decompress_gif_mp(g):
    from dmgen import gen, cores
    import multiprocessing as mp
    q_put = mp.Queue()
    q_get = mp.Queue()
    args = (q_put, q_get)
    processes = []
    with gen.timer():
        # Start the processes.
        for i in range(max(1, cores.CORES)):
            p = mp.Process(target=atomic_decompress, args=args)
            p.daemon = True
            p.start()
            processes.append(p)
        # Put the data.
        for index, data in enumerate(g.images):
            q_put.put((index, data))
        # Put the data to end the processes.
        for i in range(len(processes)):
            q_put.put(0)
        # Get the data and set on the objects.
        for i in g.images:
            index, data = q_get.get()
            g.images[index].set_decompressed_data(data)


# Multiprocess target function for decompression.
def atomic_decompress(q_get, q_put):
    while 1:
        # Get the data.
        data = q_get.get()
        # End process if no value.
        if not data:
            return
        # Run the function and return the data.
        index, gif_img = data
        gif_img.decompress_data()
        q_put.put((index, gif_img.decompressed_data))
