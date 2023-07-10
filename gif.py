import sys
import time

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


class Gif_Image(object):
    def __init__(self):
        self.comments = []
        self.graphics_extension_block = False
        self.user_input_required = None
        self.disposal_method = None
        self.frame_delay = DEFAULT_FRAME_DELAY
        self.transparent_color_index = None
        self.image_block = None
        self.x = None
        self.y = None
        self.width = None
        self.height = None
        self.lzw_data = None
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
        self.lzw_data = None

    def decompress_data(self):
        if not self.decompressed_data and self.lzw_data:
            self.set_decompressed_data(self.lzw_data.parse_stream_data())

    def process_data(self):
        '''Set additional data after decompressing.'''
        # If the data is too short, append transparent color indexes, or default to 0, until it's the right length.
        if len(self.decompressed_data) < self.width * self.height:
            if self.transparent_color_index is None:
                color_index = 0
            else:
                color_index = self.transparent_color_index
            self.decompressed_data += bytes(
                [color_index] * (self.width * self.height - len(self.decompressed_data))
            )
        # Deinterlace if needed.
        if self.interlaced:
            self.deinterlace()
        # In some gifs transparent_color_index is beyond the end of the color table, so clamp it to the end.
        if self.transparent_color_index is not None:
            if self.transparent_color_index >= len(self.color_table):
                self.transparent_color_index = len(self.color_table) - 1

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


# Counting bits starting at 0
class Gif(object):
    """class to manipulate gifs"""

    GIF87a = b'GIF87a'
    GIF89a = b'GIF89a'

    def __init__(self, filename, decompress=False, data=None):
        if VERBOSE:
            print('loading', filename)
            start_time = time.time()
        self.decompress = decompress
        self.images = []
        self.filename = filename
        if not data:
            data = open(filename, 'rb').read()
        self.data = data
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
            self.global_color_table = self.parse_color_table(global_color_table_entries)
        else:
            self.global_color_table = []

    def parse_color_table(self, table_entries):
        bytes = 3 * table_entries
        data = self.data[self.tell:self.tell + bytes]
        self.tell += bytes
        return [data[i:i+3] for i in range(0, bytes, 3)]

    def parse_blocks(self):
        while 1:
            try:
                separator = self.data[self.tell]
            # Some gifs don't include the end of file char.
            except IndexError:
                break
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
            elif separator == 0:  # Invalid, but probably a dangling terminator.
                continue
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
            self.current_image.color_table = self.parse_color_table(color_table_entries)
        else:
            # Use the global color table if there's no local one
            self.current_image.color_table = self.global_color_table
        # Find cases where the transparent index is out of bounds and force it off.
        # TODO See if there's a usual way to handle this.
        if self.current_image.transparent_color_index is not None and self.current_image.transparent_color_index > len(self.current_image.color_table):
            self.current_image.transparent_color_index = None
        self.current_image.lzw_data = self.parse_image_data()
        if self.decompress:
            self.current_image.decompress_data()

    def parse_image_data(self):
        # Make these local due to the tight loops.
        tell = self.tell
        data = self.data
        lzw_data = bytearray()
        minimum_lzw_code_size = data[tell]
        tell += 1
        while 1:
            length = data[tell]
            if not length:
                break
            # This tell usage is backwards from the norm so we can do a single assignment to self.tell.
            tell += length + 1
            lzw_data += data[tell - length:tell]
        # Re-assign to self.tell and add 1 from the length check that was just done.
        self.tell = tell + 1
        return Gif_LZW(minimum_lzw_code_size, lzw_data)

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


class Gif_LZW(object):
    # If the code table has reached the 2**12 limit, the code table may not be added to
    maximum_bit_size = 12

    bit_ands = [2 ** i - 1 for i in range(13)]

    def __init__(self, minimum_size, data):
        self.minimum_size = minimum_size
        self.clear_code = 1 << self.minimum_size
        self.end_of_information_code = self.clear_code + 1

        # The code_table will be reset to the inside of reset_code_table.
        self.code_table = []
        self.default_code_table = [None] * (1 << self.maximum_bit_size)
        self.default_code_table[:1 << minimum_size] = [bytes((i,)) for i in range(1 << minimum_size)]

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
        self.reset_code_table()
        get_next_code = self._get_next_code()
        while 1:
            response = self._parse_stream_data(get_next_code)
            # Fake tail recursion by returning end_of_information_code.
            if response == self.end_of_information_code:
                continue
            else:
                break
        return self.stream

    def _parse_stream_data(self, get_next_code):
        # Localize variable due to the loop.
        next_code_index = self.next_code_index
        code_table = self.code_table
        clear_code = self.clear_code
        end_of_information_code = self.end_of_information_code
        stream = self.stream
        # The code table is only allowed to grow to a specific size.
        table_immutable = False
        prev_code = clear_code
        # Some gifs don't respect the standard, so we don't ensure there's at least one clear code.
        # The while loop is because clear codes can appear at any time, even right after another one.
        while prev_code == clear_code:
            try:
                prev_code = next(get_next_code)
            except StopIteration:
                return
            if prev_code == end_of_information_code:
                return
        # The first code must be in the initial code table.
        stream += code_table[prev_code]
        for code in get_next_code:
            # If it's going to reference an existing code.
            if code < next_code_index:
                # Handle clear code and end of info code.
                if code == clear_code:
                    self.reset_code_table()
                    # No tail recursion, so here we are. Wipe out prev_code like this.
                    return self.end_of_information_code
                if code == end_of_information_code:
                    return
                K_code = code
            # If it's referencing a new code.
            else:
                K_code = prev_code

            # If the code table can still grow.
            if not table_immutable:
                # This is what the gif LZW algorithm does to add entries to the code table.
                # K_code depends on the above if/else block.
                code_table[next_code_index] = code_table[prev_code] + bytes((code_table[K_code][0],))
                next_code_index += 1
                # If the code index is crossing the next threshold (2**x).
                if next_code_index == self.next_code_table_grow:
                    if self.code_size == self.maximum_bit_size:
                        # Gifs aren't allowed to grow beyond this hard limit per code.
                        table_immutable = True
                    else:
                        self.set_code_size(self.code_size + 1)
            # Add to the stream.
            stream += code_table[code]
            # Set the previous code for the next loop.
            prev_code = code
        return 0

    def reset_code_table(self):
        self.code_table[:] = self.default_code_table
        # Track what the next index for a code in self.code_table will be.
        self.next_code_index = self.end_of_information_code + 1
        self.set_code_size(self.minimum_size + 1)

    def set_code_size(self, size):
        self.code_size = size
        self.next_code_table_grow = 1 << size

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

