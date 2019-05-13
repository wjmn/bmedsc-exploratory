#!/usr/bin/env python

from PIL import Image, ImageDraw, ImageFont

baseColour = (255, 255, 255)
baseSizeX, baseSizeY = (20, 20)
textColour = (0, 0, 0)

for digit in range(10):
    baseImage = Image.new("RGB", (baseSizeY, baseSizeX), baseColour)
    base = ImageDraw.Draw(baseImage)

    text = str(digit)
    textSizeX, textSizeY = base.textsize(text)
    textPosition = (baseSizeX // 2 - textSizeX // 2, baseSizeY // 2 - textSizeY // 2)
    base.text(textPosition, text, textColour)

    saveName = str(digit)
    saveExtension = "png"
    baseImage.save(f"../data/digit-images/{saveName}.{saveExtension}")
