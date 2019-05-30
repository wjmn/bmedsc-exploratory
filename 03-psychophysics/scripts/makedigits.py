#!/usr/bin/env python

from PIL import Image, ImageDraw, ImageFont

baseColour = (0, 0, 0)
baseSizeX, baseSizeY = (16, 16)
textColour = (255, 255, 255)

saveExtension = "png"

for digit in range(10):
    baseImage = Image.new("RGB", (baseSizeY, baseSizeX), baseColour)
    base = ImageDraw.Draw(baseImage)

    text = str(digit)
    textSizeX, textSizeY = base.textsize(text)
    textPosition = (baseSizeX / 2 - textSizeX / 2, baseSizeY / 2 - textSizeY / 2)
    base.text(textPosition, text, textColour)

    saveName = str(digit)
    baseImage.save(f"../data/digit-images/{saveName}.{saveExtension}")

# Blank white
baseImage = Image.new("RGB", (baseSizeY, baseSizeX), (255,255,255))
base = ImageDraw.Draw(baseImage)
baseImage.save(f'../data/digit-images/blank.{saveExtension}')
