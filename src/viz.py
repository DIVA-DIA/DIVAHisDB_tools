import matplotlib
import matplotlib.cm

from PIL import Image
import numpy as np
import os
import argparse


def main(img):
    read_img = np.asarray(Image.open(img))
    class_encodings = [0, 1, 2, 4, 8, 16]

    dest_filename = os.path.join(os.getcwd(), 'images', 'viz_' + img)
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img = np.copy(read_img)
    blue = read_img[:, :, 2]  # Extract just blue channel

    # Colours are in RGB
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = [cmap(i / len(class_encodings), bytes=True)[:3] for i in range(len(class_encodings))]

    # Get the mask for each colour
    masks = {color: (blue == i) > 0 for color, i in zip(colors, class_encodings)}

    # Color the image with relative colors
    for color, mask in masks.items():
        img[mask] = color

    # Make and save the class color encoding
    color_encoding = {str(i): color for color, i in zip(colors, class_encodings)}

    # make_colour_legend_image(os.path.join(os.path.dirname(dest_filename), "output_visualizations_colour_legend.png"),
    #                          color_encoding)

    # Write image to output folder
    Image.fromarray(img.astype(np.uint8)).save(dest_filename)


def make_colour_legend_image(img_name, colour_encoding):
    import matplotlib.pyplot as plt

    labels = sorted(colour_encoding.keys())
    colors = [tuple(np.array(colour_encoding[k])/255) for k in labels]
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", c) for c in colors]
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(img_name, dpi=1000, bbox_inches=bbox)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)

    args = parser.parse_args()

    main(**args.__dict__)
