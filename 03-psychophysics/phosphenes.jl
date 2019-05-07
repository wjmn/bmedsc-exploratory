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

## RENDERING

function render(e::Electrode{CirclePhosphene})

    xsize = 640
    ysize = 480
    pbase = 10
    
    psize = pbase * (0.5 + 4 * √((e.x-0.5)^2 + (e.y-0.5)^2)) |> floor

    base = zeros(ysize, xsize)

    xmin, xmax = max(1, (xsize*e.x)-psize), min(xsize, (xsize*e.x)+psize)
    ymin, ymax = max(1, (ysize*e.y)-psize), min(ysize, (ysize*e.y)+psize)
    base[int(xmin):int(xmax), int(ymin):int(ymax)] .= 1

    base

end


Gray.(testrender)
