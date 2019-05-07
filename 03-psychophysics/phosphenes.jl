module Phosphenes

# USING

using ImageFiltering
using Images

# TYPES

struct CirclePhosphene end
struct BarPhosphene end

Phosphene = Union{CirclePhosphene, BarPhosphene}

struct Electrode{T}
    phosphene :: T
    x :: Float64
    y :: Float64
end

# FUNCTIONS

## UTILITY

int(x) = trunc(Int64, x)

## CREATION

function regelectrodes(xsize=8, ysize=6)
    [Electrode(CirclePhosphene(), ix/xsize, iy/ysize)
        for ix in 1:xsize
        for iy in 1:ysize]
end

## RENDERING

function render(e::Electrode{CirclePhosphene})

    xsize = 640
    ysize = 480
    pbase = 3

    psize = pbase * (0.5 + 4 * √((e.x-0.5)^2 + (e.y-0.5)^2))

    base = zeros(ysize, xsize)

    xmin, xmax = max(1, (xsize*e.x)-psize), min(xsize, (xsize*e.x)+psize)
    ymin, ymax = max(1, (ysize*e.y)-psize), min(ysize, (ysize*e.y)+psize)
    base[int(ymin):int(ymax), int(xmin):int(xmax)] .= 1

    kernel = ImageFiltering.Kernel.gaussian(psize * 1.5)

    imfilter(base, kernel)

end

# TODO How to specify type of es? Array{Electrode{Phosphene}, 1} doesn't work.
function fullrender(vs::Array{Float64,1}, es)
    reduce((x,y)->x.+y, [v .* render(e) for (v,e) in zip(vs, es)])
end

end
