#!/usr/bin/env python3

import pygame
from fullgif import fit_to


class Pygame_Gif:
    def __init__(self, Gif):
        self.gif = Gif
        self.images = [Pygame_Gif_Image(image) for image in Gif.images]
    

class Pygame_Gif_Image:
    def __init__(self, Gif_Image):
        self.image = None
        self.rect = None
        self.gif_image = Gif_Image
    
    def make_pygame_surface(self):
        if self.image:
            return
        if not self.gif_image.decompressed_data:
            self.gif_image.decompress_data()
        self.gif_image.process_data()
        if self.gif_image.transparent_color_index is not None:
            # Gifs use palettes but we're using pygame colorkey for transparency,
            #  so we need to find an unused color value to apply to the surface.
            if self.gif_image.color_table.count(self.gif_image.color_table[self.gif_image.transparent_color_index]) > 1:
                for b in range(256):
                    transparent = bytes((0, 0, b))
                    if not self.gif_image.color_table.count(transparent):
                        break
                else:
                    # Only 256 possible values in the color table, so if we're here then this one can't exist in it.
                    transparent = bytes((0, 1, 0))
                # The color table could be the same object as the global color table, so make a copy.
                self.gif_image.color_table = self.gif_image.color_table[:]
                self.gif_image.color_table[self.gif_image.transparent_color_index] = transparent
        self.image = pygame.image.frombuffer(self.gif_image.decompressed_data, (self.gif_image.width, self.gif_image.height), 'P')
        self.image.set_palette(self.gif_image.color_table)
        self.image = self.image.convert(24)
        if self.gif_image.transparent_color_index is not None:
            self.image.set_colorkey(self.gif_image.color_table[self.gif_image.transparent_color_index])
        self.rect = pygame.Rect(self.gif_image.x, self.gif_image.y, self.gif_image.width, self.gif_image.height)

def display_gif(pygame_gif, fitto=(1000, 1000), loop=True):
    gif = pygame_gif.gif
    pygame.display.init()
    base_s = pygame.Surface(gif.dims)
    display_dims = fit_to(gif.dims, fitto)
    s = pygame.display.set_mode(display_dims)
    if gif.filename:
        pygame.display.set_caption(gif.filename)
    s_rect = s.get_rect()
    if gif.background_color_index:
        base_s.fill(gif.global_color_table[gif.background_color_index])
        pygame.display.flip()
    c = pygame.time.Clock()
    while 1:
        frame_delay = 0
        for i in pygame_gif.images:
            if not i.image:
                i.make_pygame_surface()
            if pygame.event.get(pygame.QUIT):
                break
            base_s.blit(i.image, i.rect)
            s.blit(pygame.transform.smoothscale(base_s, display_dims), s_rect)
            if frame_delay:
                c.tick(100. / frame_delay)
            pygame.display.flip()
            frame_delay = i.gif_image.frame_delay
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
        if not i.image:
            i.make_pygame_surface()
        pygame.image.save(i.image, os.path.join(output_folder, '%d.png' % (ind + 1)))
