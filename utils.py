def get_font_color(rgb):
    rgb_values = [int(x) for x in rgb.strip("rgb()").split(",")]
    r, g, b = [x / 255.0 for x in rgb_values]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#FFFFFF" if luminance < 0.5 else "#1F2937"