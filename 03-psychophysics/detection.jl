################################################################################
# SETUP
################################################################################

# Local modules
include("./phosphenes.jl")

# Module imports
using PyCall
using .Phosphenes: int, regelectrodes, fullrender
using Images: imresize

# PyCall imports
visual = pyimport("psychopy.visual")
core = pyimport("psychopy.core")


################################################################################
# CONSTANTS
################################################################################

# Size parameters
XSIZE = 640                 # Screen x size
YSIZE = 480                 # Screen y size
SSIZE = 50                  # Stimulus size
SCALE = 80                  # Downsampling factor for electrode grid
EXSIZE = div(XSIZE, SCALE)  # Electrode grid x size
EYSIZE = div(YSIZE, SCALE)  # Electrode grid y size


################################################################################
# FUNCTIONS
################################################################################

# Make a square stimulus in a specified location.
function makestim(xpos::Float64, ypos::Float64)
    base = zeros(YSIZE, XSIZE)
    ymin, ymax = max(1, ypos*YSIZE-SSIZE), min(YSIZE, ypos*YSIZE+SSIZE)
    xmin, xmax = max(1, xpos*XSIZE-SSIZE), min(XSIZE, xpos*XSIZE+SSIZE)

    # # CROSS
    # base[int(ymin):int(ymax), int(xpos*XSIZE)] .= 1
    # base[int(ypos*YSIZE), int(xmin):int(xmax)] .= 1

    base[int(ymin):int(ymax), int(xmin):int(xmax)] .= 1

    base
end

# Process a stimulus into a vector of values
function procstim(stimulus :: Array{Float64, 2})
    downsampled = imresize(stimulus, (EYSIZE, EXSIZE))
    flattened = vec(downsampled)
    flattened
end


################################################################################
# PSYCHOPY SCRIPTING
################################################################################

# Make window
window = visual.Window()

# Initialise electrodes
electrodes = regelectrodes(EXSIZE, EYSIZE)

# Loop
for i in 1:100
    xpos, ypos = rand(Float64, 2)

    stimulus = makestim(xpos, ypos)
    processed = procstim(stimulus)
    rendered = fullrender(processed, electrodes)

    imstim = visual.ImageStim(window, image=rendered)
    imstim.draw()
    window.flip()
end
